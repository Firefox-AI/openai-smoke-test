import json
import time
import logging
from pathlib import Path
import typer
from datetime import datetime, timezone
from fxa.core import Client
from fxa.tests.utils import TestEmailAccount
from fxa.tools.bearer import get_bearer_token
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn

app = typer.Typer(add_completion=False)

PASSWORD = "123dev123dev123dev"
CLIENT_ID = "5882386c6d801776"
USERS_FILE = Path("users.json")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def get_env_urls(env: str):
    if env == "stage":
        return (
            "https://api-accounts.stage.mozaws.net/v1",
            "https://oauth.stage.mozaws.net",
        )
    elif env == "prod":
        return "https://api.accounts.firefox.com", "https://oauth.accounts.firefox.com"
    else:
        raise ValueError("env must be 'prod' or 'stage'")


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))
    logging.info(f"Saved {path}")


@app.command("create-tokens")
def create_tokens(
    n_users: int = typer.Option(..., "--n-users"),
    env: str = typer.Option(..., "--env", help="Environment: prod or stage"),
):
    fxa_base, oauth_base = get_env_urls(env)
    client = Client(fxa_base)
    users = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Creating FxA test users", total=n_users)
        for _ in range(n_users):
            acct = TestEmailAccount()
            session = client.create_account(acct.email, PASSWORD)
            time.sleep(1)
            acct.fetch()
            for m in acct.messages:
                code = m["headers"].get("x-verify-code")
                if code:
                    session.verify_email_code(code)
                    break
            session = client.login(acct.email, PASSWORD)
            if not session.verified:
                progress.advance(task)
                continue
            try:
                token = get_bearer_token(
                    acct.email,
                    PASSWORD,
                    scopes=["profile"],
                    client_id=CLIENT_ID,
                    account_server_url=fxa_base.replace("/v1", ""),
                    oauth_server_url=oauth_base,
                )
                users.append(
                    {
                        "email": acct.email,
                        "password": PASSWORD,
                        "token": token,
                        "refreshed_at": datetime.now(timezone.utc).isoformat(),
                        "env": env,
                    }
                )
            except Exception:
                pass
            acct.clear()
            progress.advance(task)

    save_json(USERS_FILE, users)
    logging.info(f"Created {len(users)} users and tokens")


@app.command("refresh-tokens")
def refresh_tokens():
    if not USERS_FILE.exists():
        logging.error("users.json not found")
        raise typer.Exit(1)

    users = json.loads(USERS_FILE.read_text())

    if not users:
        logging.error("users.json is empty")
        raise typer.Exit(1)

    env = users[0].get("env")
    if env not in {"prod", "stage"}:
        logging.error("invalid or missing env in users.json")
        raise typer.Exit(1)

    fxa_base, oauth_base = get_env_urls(env)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Refreshing tokens", total=len(users))
        for u in users:
            try:
                token = get_bearer_token(
                    u["email"],
                    u["password"],
                    scopes=["profile"],
                    client_id=CLIENT_ID,
                    account_server_url=fxa_base.replace("/v1", ""),
                    oauth_server_url=oauth_base,
                )
                u["token"] = token
                u["refreshed_at"] = datetime.now(timezone.utc).isoformat()
            except Exception:
                pass
            progress.advance(task)

    save_json(USERS_FILE, users)
    logging.info(f"Refreshed {len(users)} tokens for env={env}")


if __name__ == "__main__":
    app()
