# services/llm_service.py
import requests
from typing import Dict, Any
from services.config import LLM_MODEL_NAME, LLM_BASE_URL, REQUEST_TIMEOUT

class LLMService:
    def __init__(self, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL_NAME) -> None:
        self.base_url: str = base_url
        self.model: str = model
        self.timeout: int = REQUEST_TIMEOUT
   
    def chat(self, prompt: str) -> Dict[str, Any]:
        if not prompt.strip():
            return {"error": "Empty prompt", "status": "error"}
            
        try:
            response = requests.post(
                f'{self.base_url}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result and result['response']:
                    return {"answer": result['response'], "status": "success"}
                else:
                    return {"error": "Empty response from LLM", "status": "error"}
            else:
                return {"error": f"LLM error: {response.status_code}", "status": "error"}
                
        except requests.exceptions.Timeout:
            return {"error": "LLM request timed out", "status": "error"}
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to LLM. Is Ollama running?", "status": "error"}
        except Exception as e:
            return {"error": str(e), "status": "error"}