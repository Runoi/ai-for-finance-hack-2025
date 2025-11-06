"""
Модуль, реализующий "думающих" агентов для FAIR-RAG цикла.
"""
import json
import re
from typing import Optional
from .llm_client import LLMClient
from utils.prompt_library import PROMPT_LIBRARY
from config import Config, config

class SeaAgent:
    """Агент-Аналитик для аудита доказательств (Structured Evidence Assessment)."""
    def __init__(self, llm_client: LLMClient, config_override: Optional[Config] = None):
        self.llm_client = llm_client
        self.prompt_template = PROMPT_LIBRARY["sea_agent"]
        self.config = config_override or config

    def analyze(self, question: str, context: str) -> dict:
        """
        Анализирует контекст и возвращает отчет о достаточности информации.
        Теперь с "пуленепробиваемой" обработкой ошибок.
        """
        prompt = self.prompt_template.format(question=question, context=context)
        try:
            response_str = self.llm_client.generate(prompt, model_name=self.config.ANALYST_MODEL)
            
            # Ищем JSON-блок, который начинается с `{` и заканчивается `}`
            match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if match:
                json_str = match.group(0)
                report = json.loads(json_str)
                # Проверяем наличие ключевых полей для надежности
                if "is_sufficient" in report and "remaining_gaps" in report:
                    return report
                else:
                    raise KeyError("В JSON отчете отсутствуют обязательные ключи.")
            else:
                raise json.JSONDecodeError("JSON-объект не найден в ответе модели.", response_str, 0)

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # --- ИСПРАВЛЕННЫЙ FALLBACK ---
            # Если агент не справился, мы не можем доверять его анализу.
            # Поэтому мы считаем, что информации НЕ достаточно, и в качестве
            # "пробела" просим уточнить ИСХОДНЫЙ ВОПРОС.
            # Это гарантирует, что `RefinementAgent` получит задачу и цикл продолжится.
            print(f"!!! Ошибка или некорректный формат ответа от SeaAgent: {e}. Запускаю 2-ю итерацию с исходным вопросом.")
            return {
                "is_sufficient": "No", 
                "analysis_summary": "Агент-аналитик не смог обработать контекст. Требуется повторный, более глубокий поиск.", 
                "confirmed_findings": [],
                "remaining_gaps": [question] # <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
            }

class RefinementAgent:
    """Агент-Поисковик для генерации уточняющих запросов."""
    def __init__(self, llm_client: LLMClient, config_override: Optional[Config] = None):
        self.llm_client = llm_client
        self.prompt_template = PROMPT_LIBRARY["refinement_agent"]
        self.config = config_override or config

    def refine(self, question: str, analysis_summary: str, previous_queries: list) -> list:
        """Генерирует новые запросы на основе анализа пробелов."""
        prompt = self.prompt_template.format(
            question=question,
            analysis_summary=analysis_summary,
            previous_queries="\n".join(f"- {q}" for q in previous_queries)
        )
        try:
            response_str = self.llm_client.generate(prompt, model_name=self.config.REFINER_MODEL)
            new_queries = json.loads(response_str)
            return new_queries if isinstance(new_queries, list) else []
        except (json.JSONDecodeError, TypeError) as e:
            print(f"!!! Ошибка парсинга ответа от RefinementAgent: {e}")
            return []
        
class DecompositionAgent:
    """Агент для первичной декомпозиции сложного вопроса."""
    def __init__(self, llm_client: LLMClient, config_override: Optional[Config] = None):
        self.llm_client = llm_client
        self.prompt_template = PROMPT_LIBRARY["query_decomposer"]
        self.config = config_override or config
    def decompose(self, question: str) -> list[str]:
        """Разбивает исходный вопрос на список под-вопросов."""
        prompt = self.prompt_template.format(question=question)
        try:
            response_str = self.llm_client.generate(prompt, model_name=config.DECOMPOSER_MODEL)
            sub_queries = json.loads(response_str)
            return sub_queries if isinstance(sub_queries, list) else [question]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"!!! Ошибка парсинга ответа от DecompositionAgent: {e}. Используем исходный вопрос.")
            return [question]