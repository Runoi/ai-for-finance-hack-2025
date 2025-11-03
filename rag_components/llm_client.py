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
import tiktoken
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv
from config import config
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

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("gpt2") # Запасной вариант

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def generate(self, prompt: str, model_name: str) -> str:
        """Генерирует текстовый ответ от указанной LLM."""
        messages = [{"role": "user", "content": prompt}]
        # --- 1. СЧИТАЕМ ТОКЕНЫ НА ВХОДЕ ---
        prompt_tokens = len(self.tokenizer.encode(prompt))

        try:
            response = litellm.completion(
                model=model_name,
                messages=messages,
                temperature=0.1,
                timeout=120,
                use_litellm_proxy=True
            )
            
            # Безопасный доступ к ответу
            answer_content = ""
            choices = getattr(response, 'choices', [])
            if choices:
                first_choice = choices[0]
                message = getattr(first_choice, 'message', None)
                if message:
                    answer_content = getattr(message, 'content', "")

            # --- 2. СЧИТАЕМ ТОКЕНЫ НА ВЫХОДЕ И СТОИМОСТЬ ---
            completion_tokens = len(self.tokenizer.encode(answer_content))
            
            prices = config.MODEL_PRICES.get(model_name)
            if prices:
                cost = (
                    (prompt_tokens / 1_000_000) * prices["input"] +
                    (completion_tokens / 1_000_000) * prices["output"]
                )
                if cost > 0:
                    resource_manager.log_api_call(
                        model_name, cost, prompt_tokens, completion_tokens
                    )

            return answer_content or "Ошибка: получен пустой ответ от модели."
            
        except Exception as e:
            print(f"!!! Ошибка API при генерации ({model_name}): {e}. Повторная попытка...")
            raise
            
        
    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def get_embeddings(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Получает эмбеддинги и самостоятельно рассчитывает стоимость."""
        texts = [text.replace("\n", " ") for text in texts if text.strip()]
        if not texts:
            return []
        
        # --- 1. СЧИТАЕМ ТОКЕНЫ НА ВХОДЕ ---
        total_tokens = sum(len(self.tokenizer.encode(text)) for text in texts)
            
        try:
            response = self.embedding_openai_client.embeddings.create(
                model=model_name,
                input=texts
            )
            
            # --- 2. СЧИТАЕМ СТОИМОСТЬ ---
            prices = config.MODEL_PRICES.get(model_name)
            if prices:
                cost = (total_tokens / 1_000_000) * prices["input"]
                if cost > 0:
                    resource_manager.log_api_call(
                        model_name, cost, prompt_tokens=total_tokens
                    )
            
            return [data.embedding for data in response.data] if response.data else []
        
        except Exception as e:
            print(f"!!! Ошибка API при создании эмбеддингов ({model_name}): {e}. Повторная попытка...")
            raise