"""
Модуль для инкапсуляции всей логики взаимодействия с LLM и Embedding API.

Этот клиент использует:
- `litellm`: для унифицированного вызова различных моделей через единый интерфейс.
- `tenacity`: для автоматических повторных попыток при сбоях API, что делает
  систему более устойчивой к временным сетевым проблемам.
- `dotenv`: для безопасной загрузки API-ключей из `.env` файла.
- `ResourceManager`: для интеграции с нашей системой мониторинга и логирования затрат.
"""

import os
from typing import List, cast, Optional
import litellm
from litellm.files.main import ModelResponse
from litellm import completion
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv

from utils.resource_manager import resource_manager

load_dotenv()

class LLMClient:
    """
    Класс-клиент для взаимодействия с API, использующий SecretStr для безопасного
    хранения ключей.
    """
    def __init__(self):
        """
        Инициализирует клиент, загружая API-ключи и оборачивая их в SecretStr.
        """
        gen_key = os.getenv("LLM_API_KEY")
        embed_key = os.getenv("EMBEDDER_API_KEY")

        if not gen_key or not embed_key:
            raise ValueError("API ключи не найдены в .env файле.")

        self.gen_api_key: str = str(gen_key)
        self.embed_api_key: str = str(embed_key)

        self.base_url = "https://ai-for-finance-hack.up.railway.app/"
        litellm.api_base = self.base_url
        litellm.api_key = self.gen_api_key
        litellm.use_litellm_proxy = True

        self.embedding_openai_client = OpenAI(
            api_key=self.embed_api_key,
            base_url=self.base_url
        )

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def generate(self, prompt: str, model_name: str) -> str:
        """Генерирует текстовый ответ от указанной LLM."""
        messages = [{"role": "user", "content": prompt}]
        try:
            
            response = litellm.completion(
                model=model_name,
                messages=messages,
                temperature=0.1,
                timeout=120
            )
            
            response = cast(ModelResponse, response)
            cost = getattr(response, 'usage', {}).get('total_cost', 0.0)
            if cost is not None:
                resource_manager.log_api_call(model_name, cost)

            # 3. ТОТАЛЬНО БЕЗОПАСНЫЙ доступ к ответу
            answer_content = ""
            choices = getattr(response, 'choices', [])
            if choices:
                first_choice = choices[0]
                message = getattr(first_choice, 'message', None)
                if message:
                    answer_content = getattr(message, 'content', "")

            if answer_content:
                return answer_content
            else:
                print("!!! WARNING: Получен пустой или некорректный ответ от API.")
                return "Ошибка: получен пустой ответ от модели."
            

        except Exception as e:
            print(f"!!! Ошибка API при генерации ({model_name}): {e}. Повторная попытка...")
            raise
        
    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def get_embeddings(self, texts: List[str], model_name: str) -> List[List[float]]:
        """
        Получает эмбеддинги, ИСПОЛЬЗУЯ ПРЯМОЙ ВЫЗОВ `openai` SDK.
        """
        texts = [text.replace("\n", " ") for text in texts if text.strip()]
        if not texts:
            return []
            
        try:
            
            response = self.embedding_openai_client.embeddings.create(
                model=model_name,
                input=texts
            )
            
            #Подсчет стоимости для OpenAI эмбеддингов (примерный)
            total_tokens = response.usage.total_tokens
            cost = (total_tokens / 1_000_000) * 0.02 # Цена для text-embedding-3-small
            resource_manager.log_api_call(model_name, cost)
            
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"!!! Ошибка API при создании эмбеддингов ({model_name}): {e}. Повторная попытка...")
            raise