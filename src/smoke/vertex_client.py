import time
import google.auth
from google.oauth2 import service_account
import google.auth.transport.requests
import aiohttp
import certifi
import ssl

REQUIRED_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

class VertexClient:
	def __init__(self, vertex_region, vertex_uri, summarization_config):
		self.vertex_region = vertex_region
		self.vertex_uri = vertex_uri
		self.config = summarization_config
		credentials_path = self.config.get("service_account_file", "creds.json")
		self.creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=REQUIRED_SCOPES)

		self._vertex_ai_token = None
		self._vertex_ai_token_refresh_time = 0
		self.ssl_context = ssl.create_default_context(cafile=certifi.where())

	@property
	def vertex_ai_token(self):
		"""A property that returns a valid access token, refreshing it if necessary."""
		if time.time() - self._vertex_ai_token_refresh_time > 300:
			print("Refreshing Vertex AI access token from service account...")
			self._refresh_vertex_ai_token()

		return self._vertex_ai_token

	def _refresh_vertex_ai_token(self):
		"""Uses the loaded service account credentials to get a new token."""
		try:
			# The credentials object handles its own refreshing
			auth_req = google.auth.transport.requests.Request()
			self.creds.refresh(auth_req)

			self._vertex_ai_token = self.creds.token
			self._vertex_ai_token_refresh_time = time.time()
			print("Token refreshed successfully.")

		except Exception as e:
			print(f"Error refreshing token: {e}")
			raise

	async def vertex_completion(self, payload):
		headers = {
			"Authorization": f"Bearer {self.vertex_ai_token}",
			"Accept": "application/json"
		}
		url = f"https://{self.vertex_region}-aiplatform.googleapis.com/v1/{self.vertex_uri}:predict"
		first_token_time = None
		async with aiohttp.ClientSession() as session:
			async with session.post(url=url, json=payload, headers=headers, ssl=self.ssl_context) as response:
				if response.status != 200:
					print(f"Request failed with status code: {response.status}")
					return "", first_token_time, {}

				try:
					response_dict = await response.json()
					summary = response_dict["predictions"][0]["text"]
					input_tokens = response_dict["predictions"][0]["meta_info"]["prompt_tokens"]
					completion_tokens = response_dict["predictions"][0]["meta_info"]["completion_tokens"]
				except aiohttp.ContentTypeError as e:
					print(f"Error decoding JSON: {e}")
					print(f"Raw response: {await response.text()}")
					summary = ""
				return summary, first_token_time, {
					"input_tokens": input_tokens,
					"completion_tokens": completion_tokens
				}
