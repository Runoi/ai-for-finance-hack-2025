"""
Модуль, реализующий "думающих" агентов для FAIR-RAG цикла.
"""
import json
from .llm_client import LLMClient
from utils.prompt_library import PROMPT_LIBRARY
from config import config

class SeaAgent:
    """Агент-Аналитик для аудита доказательств (Structured Evidence Assessment)."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_template = PROMPT_LIBRARY["sea_agent"]

    def analyze(self, question: str, context: str) -> dict:
        """Анализирует контекст и возвращает отчет о достаточности информации."""
        prompt = self.prompt_template.format(question=question, context=context)
        try:
            response_str = self.llm_client.generate(prompt, model_name=config.ANALYST_MODEL)
            report = json.loads(response_str)
            return report
        except (json.JSONDecodeError, KeyError) as e:
            print(f"!!! Ошибка парсинга ответа от SeaAgent: {e}")
            return {"is_sufficient": "No", "analysis_summary": "Ошибка анализа.", "remaining_gaps": []}

class RefinementAgent:
    """Агент-Поисковик для генерации уточняющих запросов."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_template = PROMPT_LIBRARY["refinement_agent"]

    def refine(self, question: str, analysis_summary: str, previous_queries: list) -> list:
        """Генерирует новые запросы на основе анализа пробелов."""
        prompt = self.prompt_template.format(
            question=question,
            analysis_summary=analysis_summary,
            previous_queries="\n".join(f"- {q}" for q in previous_queries)
        )
        try:
            response_str = self.llm_client.generate(prompt, model_name=config.REFINER_MODEL)
            new_queries = json.loads(response_str)
            return new_queries if isinstance(new_queries, list) else []
        except (json.JSONDecodeError, TypeError) as e:
            print(f"!!! Ошибка парсинга ответа от RefinementAgent: {e}")
            return []