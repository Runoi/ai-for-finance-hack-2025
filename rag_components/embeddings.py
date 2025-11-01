from typing import List
from langchain_core.embeddings import Embeddings
from .llm_client import LLMClient
from config import config

class LiteLLMEmbeddings(Embeddings):
    """
    Кастомная реализация интерфейса эмбеддера LangChain,
    использующая наш унифицированный LLMClient с litellm.
    """
    def __init__(self, client: LLMClient):
        self.client = client
        self.model_name = config.EMBEDDING_MODEL

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Метод для векторизации списка документов.
        LangChain вызывает именно его.
        """
        return self.client.get_embeddings(texts=texts, model_name=self.model_name)

    def embed_query(self, text: str) -> List[float]:
        """
        Метод для векторизации одного поискового запроса.
        LangChain вызывает именно его при поиске.
        """
        result = self.client.get_embeddings(texts=[text], model_name=self.model_name)
        return result[0] if result else []