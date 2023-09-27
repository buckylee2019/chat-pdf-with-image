from ibm_cloud_sdk_core import IAMTokenManager
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, BearerTokenAuthenticator

class Credentials:
    DEFAULT_API = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"

    def __init__(
        self,
        api_key: str,
        project_id: str,
        api_endpoint: str = DEFAULT_API,
    ):
        """
        Instansiate the credentials object

        Args:
            api_key (str): The GENAI API Key
            api_endpoint (str, optional): GENAI API Endpoint. Defaults to DEFAULT_API.
        """
        if api_key is None:
            raise ValueError("api_key must be provided")
        self.api_key = api_key
        if api_endpoint is None:
            raise ValueError("api_endpoint must be provided")
        self.api_endpoint = api_endpoint
        if project_id is None:
            raise ValueError("project_id must be provided")
        self.project_id = project_id

        # New dict for ibm-watson-machine-learning-sdk
        self.wml_credentials = {
            "apikey": self.api_key,
            "url": self.api_endpoint
        }