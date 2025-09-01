# services/llm_service.py
import requests
import logging
from typing import Dict, Any

from services.config import settings

logger = logging.getLogger(settings.LOGGER_NAME)

class LLMService:
    """A service to interact with a local LLM API (e.g., Ollama)."""
    
    def __init__(self, base_url: str, model: str, timeout: int = settings.REQUEST_TIMEOUT):
        """
        Initializes the LLMService.

        Args:
            base_url: The base URL of the LLM API.
            model: The name of the model to use.
            timeout: The request timeout in seconds.
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
   
    def chat(self, prompt: str) -> Dict[str, Any]:
        """
        Sends a prompt to the LLM and returns the response.
        """
        if not prompt or not prompt.strip():
            logger.warning("LLM chat called with an empty prompt.")
            return {"error": "Empty prompt provided", "status": "error"}
            
        try:
            logger.info(f"Sending prompt to LLM model '{self.model}'...")
            response = requests.post(
                f'{self.base_url}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            if 'response' in result and result['response']:
                logger.info("Successfully received response from LLM.")
                return {"answer": result['response'].strip(), "status": "success"}
            else:
                logger.error("LLM response was empty or malformed.")
                return {"error": "Empty response from LLM", "status": "error"}
                
        except requests.exceptions.Timeout:
            logger.error(f"LLM request timed out after {self.timeout} seconds.")
            return {"error": "LLM request timed out", "status": "error"}
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to LLM at {self.base_url}. Is the service running?")
            return {"error": "Cannot connect to LLM service", "status": "error"}
        except requests.exceptions.HTTPError as e:
            logger.error(f"LLM service returned an error: {e.response.status_code} {e.response.text}")
            return {"error": f"LLM error: {e.response.status_code}", "status": "error"}
        except Exception as e:
            logger.error(f"An unexpected error occurred in LLMService: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}