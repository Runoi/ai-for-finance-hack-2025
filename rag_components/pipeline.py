"""
Главный класс-оркестратор `RAGPipeline`, реализующий переключаемые стратегии
обработки запросов.
"""

from typing import List, Optional

# --- Импорты LangChain ---
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# --- Импорты наших модулей ---
from config import Config, config
from utils.resource_manager import resource_manager
from utils.prompt_library import PROMPT_LIBRARY
from rag_components.llm_client import LLMClient
from rag_components.agents import SeaAgent, RefinementAgent, DecompositionAgent
# Импортируем реранкер, но пока не будем его требовать в __init__
# from rag_components.reranker import APIReranker


class RAGPipeline:
    """
    Основной класс-оркестратор. Выбирает и запускает одну из трех
    стратегий обработки в зависимости от `config.STRATEGY`.
    """
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client: LLMClient,
        sea_agent: SeaAgent,
        refinement_agent: RefinementAgent,
        decomposition_agent: DecompositionAgent,
        # reranker: Optional[APIReranker] = None, # Заглушка для будущего
        config_override: Optional[Config] = None,
    ):
        self.retriever = retriever
        # self.reranker = reranker
        self.llm_client = llm_client
        self.sea_agent = sea_agent
        self.refinement_agent = refinement_agent
        self.decomposition_agent = decomposition_agent
        self.config = config_override or config

    def run(self, question: str) -> str:
        """
        Главный метод. Выбирает и запускает стратегию на основе `self.config.STRATEGY`.
        """
        resource_manager.log_checkpoint(f"Старт обработки. Стратегия: {self.config.STRATEGY}")
        
        if self.config.STRATEGY == 'DECOMPOSE':
            return self._run_decomposition_strategy(question)
        elif self.config.STRATEGY == 'SIMPLE':
            return self._run_simple_strategy(question)
        else: # По умолчанию 'ITERATIVE'
            return self._run_iterative_strategy(question)

    def _run_simple_strategy(self, question: str) -> str:
        """Стратегия №3: Простой RAG. Поиск -> Генерация. (1 LLM-вызов)"""
        resource_manager.log_checkpoint("Выбран путь: Simple RAG")
        
        # 1. Поиск (без реранкинга)
        docs = self.retriever.invoke(question)
        docs_for_context = docs[:self.config.MAX_CONTEXT_DOCS]

        if not docs_for_context:
            return "К сожалению, по вашему запросу не удалось найти релевантную информацию."

        # 2. Финальная Генерация
        final_context = "\n\n".join([doc.page_content for doc in docs_for_context])
        final_prompt = PROMPT_LIBRARY["final_generator"].format(context=final_context, question=question)
        answer = self.llm_client.generate(final_prompt, model_name=self.config.GENERATOR_MODEL)
        
        resource_manager.log_checkpoint("Завершение обработки (Simple)")
        return answer

    def _run_decomposition_strategy(self, question: str) -> str:
        """Стратегия №1: Сначала разбить, потом искать. (2 LLM-вызова)"""
        resource_manager.log_checkpoint("Выбран путь: Decomposition-First")
        
        # 1. Декомпозиция
        sub_queries = self.decomposition_agent.decompose(question)
        
        # 2. Поиск
        all_docs: List[Document] = []
        for q in sub_queries:
            all_docs.extend(self.retriever.invoke(q))
        
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        docs_for_context = unique_docs[:self.config.MAX_CONTEXT_DOCS]

        if not docs_for_context:
             return "К сожалению, по вашему запросу не удалось найти релевантную информацию."
        
        # 3. Финальная Генерация
        final_context = "\n\n".join([doc.page_content for doc in docs_for_context])
        final_prompt = PROMPT_LIBRARY["final_generator"].format(context=final_context, question=question)
        answer = self.llm_client.generate(final_prompt, model_name=self.config.GENERATOR_MODEL)
        
        resource_manager.log_checkpoint("Завершение обработки (Decomposition)")
        return answer

    def _run_iterative_strategy(self, question: str) -> str:
        """Стратегия №2: Искать, потом анализировать."""
        resource_manager.log_checkpoint("Выбран путь: Iterative-Refinement")
        
        collected_docs: List[Document] = []
        current_queries: List[str] = [question]
        analysis_summary = "Первоначальный запрос пользователя."

        for i in range(self.config.MAX_ITERATIONS):
            iteration = i + 1
            resource_manager.log_checkpoint(f"Старт итерации {iteration}/{self.config.MAX_ITERATIONS}")
            
            # Поиск и сбор
            new_docs_this_iteration = []
            for q in current_queries:
                retrieved = self.retriever.invoke(q)
                new_docs_this_iteration.extend(retrieved)

            seen_contents = {doc.page_content for doc in collected_docs}
            unique_new_docs = [doc for doc in new_docs_this_iteration if doc.page_content not in seen_contents]
            collected_docs.extend(unique_new_docs)
            
            if not collected_docs:
                resource_manager.log_checkpoint("Документы не найдены, прерываем цикл")
                break

            # Обрезка контекста
            docs_for_context = collected_docs[:self.config.MAX_CONTEXT_DOCS]
            context_str = "\n\n".join([doc.page_content for doc in docs_for_context])
            resource_manager.log_checkpoint(f"Используется {len(docs_for_context)} док-ов для контекста")

            # Аудит (SEA)
            report = self.sea_agent.analyze(question, context_str)
            
            if not report:
                resource_manager.log_checkpoint("Ошибка SEA-агента, прерываем цикл")
                break
                
            analysis_summary = report.get("analysis_summary", "Анализ не удался.")
            resource_manager.log_checkpoint(f"SEA-агент решил: Достаточно? -> {report.get('is_sufficient')}")

            if report.get("is_sufficient") == "Yes":
                break
            
            # Уточнение (Refinement)
            if iteration < self.config.MAX_ITERATIONS and report.get("remaining_gaps"):
                new_queries = self.refinement_agent.refine(question, analysis_summary, current_queries)
                if new_queries:
                    current_queries = new_queries
                else:
                    break
            else:
                 break

        # Финальная Генерация
        resource_manager.log_checkpoint("Начало финальной генерации ответа")
        if not collected_docs:
            return "К сожалению, по вашему запросу не удалось найти релевантную информацию в базе знаний."

        final_docs_for_generation = collected_docs[:self.config.MAX_CONTEXT_DOCS]
        final_context = "\n\n".join([doc.page_content for doc in final_docs_for_generation])
        
        final_prompt = PROMPT_LIBRARY["final_generator"].format(context=final_context, question=question)
        answer = self.llm_client.generate(prompt=final_prompt, model_name=self.config.GENERATOR_MODEL)
        
        resource_manager.log_checkpoint("Завершение обработки вопроса (Iterative)")
        return answer