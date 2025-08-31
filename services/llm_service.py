# services/llm_service.py

import requests
from typing import Dict, Any, Union

from services.config import LLM_MODEL_NAME, LLM_BASE_URL

class LLMService:
   def __init__(self, base_url = LLM_BASE_URL, model = LLM_MODEL_NAME) -> None:
       self.base_url: str = base_url
       self.model: str = model
   
   def chat(self, prompt: str) -> Dict[str, Any]:
       try:
           response: requests.Response = requests.post(f'{self.base_url}/api/generate',
                                  json={
                                      'model': self.model,
                                      'prompt': prompt,
                                      'stream': False
                                  })
           
           if response.status_code == 200:
               result: Dict[str, Any] = response.json()
               return {"answer": result['response'], "status": "success"}
           else:
               return {"error": f"Ollama error: {response.status_code}", "status": "error"}
       except Exception as e:
           return {"error": str(e), "status": "error"}