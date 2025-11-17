import json
import sys
import time
import logging
from pathlib import Path
import typer
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
import hashlib


QA_CERT_DIR = Path(__file__).parent.resolve() / "qa_certificates"


def generate_key_id_from_ec_public_key(
    public_key: ec.EllipticCurvePublicKey,
) -> tuple[bytes, str, str]:
    """Generate key_id from an EC public key using uncompressed point format

    Args:
        public_key: Elliptic curve public key (SECP256R1)

    Returns:
        tuple: (key_id_bytes, key_id_hex, key_id_b64)
    """
    # Get uncompressed point format (0x04 + X + Y, 65 bytes total)
    pubkey_uncompressed = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    # Hash the uncompressed public key to create key_id
    key_id_bytes = hashlib.sha256(pubkey_uncompressed).digest()
    # Base64 encode the raw bytes (not the hex string)
    key_id_b64 = base64.urlsafe_b64encode(key_id_bytes).decode("utf-8")
    return key_id_bytes, key_id_b64


app = typer.Typer(add_completion=False)
USERS_FILE = Path(__file__).parent.resolve() / "users.json"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))
    logging.info(f"Saved {path}")


@app.command("create-users")
def create_users(
    n_users: int = typer.Option(..., "--n-users"),
):
    users = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Creating appattest test users", total=n_users)
        for _ in range(n_users):
            time.sleep(1)
            device_private_key = ec.generate_private_key(ec.SECP256R1())
            device_public_key = device_private_key.public_key()

            _, key_id_b64 = generate_key_id_from_ec_public_key(device_public_key)

            # Store the device private key (PEM format) for use in attestation generation
            device_private_key_pem = device_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            # Store device public key in uncompressed format for reference
            pubkey_uncompressed = device_public_key.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )

            # Store key_id info as JSON (all formats for convenience)
            key_id_info = {
                "key_id_b64": key_id_b64,  # For API calls (base64 of raw bytes)
                "device_private_key_pem": device_private_key_pem.decode(
                    "utf-8"
                ),  # Device private key for attestation generation
                "device_public_key_uncompressed_b64": base64.b64encode(
                    pubkey_uncompressed
                ).decode("utf-8"),  # Uncompressed public key
            }
            users.append(key_id_info)
            progress.advance(task)

    save_json(USERS_FILE, users)
    logging.info(f"Created {len(users)} users and tokens")


if __name__ == "__main__":
    app()
