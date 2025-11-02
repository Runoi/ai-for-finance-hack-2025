# rag_components/pipeline.py
"""
Главный класс-оркестратор `RAGPipeline`, реализующий агентский цикл FAIR-RAG.

Этот пайплайн управляет всем процессом ответа на вопрос: от итеративного
поиска и анализа информации до генерации финального, основанного на фактах, ответа.
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
from rag_components.agents import SeaAgent, RefinementAgent


class RAGPipeline:
    """
    Основной класс-оркестратор, управляющий всем RAG-процессом.

    Инициализируется всеми необходимыми компонентами (Dependency Injection)
    и имеет один главный метод `run` для обработки одного вопроса.
    """
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client: LLMClient,
        sea_agent: SeaAgent,
        refinement_agent: RefinementAgent,
        config_override: Optional[Config] = None,
    ):
        """
        Инициализирует пайплайн.

        Args:
            retriever (BaseRetriever): Собранный ансамблевый ретривер.
            llm_client (LLMClient): Клиент для взаимодействия с LLM API.
            sea_agent (SeaAgent): Агент-аналитик для оценки полноты информации.
            refinement_agent (RefinementAgent): Агент для генерации уточняющих запросов.
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.sea_agent = sea_agent
        self.refinement_agent = refinement_agent
        self.config = config_override or config

    def run(self, question: str) -> str:
        """
        Основной метод, запускающий RAG-пайплайн для одного вопроса.

        Реализует итеративный цикл FAIR-RAG с ограничением контекста
        для предотвращения превышения лимитов LLM.
        """
        resource_manager.log_checkpoint(f"Начало обработки вопроса: '{question[:30]}...'")

        collected_docs: List[Document] = []
        current_queries: List[str] = [question]
        analysis_summary = "Первоначальный запрос пользователя."

        for i in range(self.config.MAX_ITERATIONS):
            iteration = i + 1
            resource_manager.log_checkpoint(f"Старт итерации {iteration}/{self.config.MAX_ITERATIONS}")
            
            # --- Шаг 1: Поиск и сбор ---
            new_docs_this_iteration = []
            for q in current_queries:
                # Мы по-прежнему ищем по всем запросам, чтобы собрать как можно больше кандидатов
                retrieved = self.retriever.invoke(q)
                new_docs_this_iteration.extend(retrieved)

            # Дедупликация. collected_docs теперь содержит всех уникальных кандидатов.
            seen_contents = {doc.page_content for doc in collected_docs}
            unique_new_docs = [doc for doc in new_docs_this_iteration if doc.page_content not in seen_contents]
            collected_docs.extend(unique_new_docs)
            resource_manager.log_checkpoint(f"Собрано {len(collected_docs)} всего уник. документов")
            
            if not collected_docs:
                resource_manager.log_checkpoint("Документы не найдены, прерываем цикл")
                break

            # --- ИЗМЕНЕНИЕ: Обрезка контекста перед анализом ---
            # Берем только N самых релевантных документов (которые находятся в начале списка)
            docs_for_context = collected_docs[:self.config.MAX_CONTEXT_DOCS]
            context_str = "\n\n".join([doc.page_content for doc in docs_for_context])
            resource_manager.log_checkpoint(f"Используется {len(docs_for_context)} док-ов для контекста")


            # --- Шаг 2: Аудит Доказательств (SEA) на ОБРЕЗАННОМ контексте ---
            report = self.sea_agent.analyze(question, context_str)
            
            if not report:
                resource_manager.log_checkpoint("Ошибка SEA-агента, прерываем цикл")
                break
                
            analysis_summary = report.get("analysis_summary", "Анализ не удался.")
            resource_manager.log_checkpoint(f"SEA-агент решил: Достаточно? -> {report.get('is_sufficient')}")

            # --- Шаг 3: Проверка Достаточности ---
            if report.get("is_sufficient") == "Yes":
                resource_manager.log_checkpoint("Информации достаточно, завершаем цикл")
                break
            
            # --- Шаг 4: Уточнение Запросов (если нужно) ---
            if iteration < self.config.MAX_ITERATIONS and report.get("remaining_gaps"):
                new_queries = self.refinement_agent.refine(question, analysis_summary, current_queries)
                if new_queries:
                    current_queries = new_queries
                    resource_manager.log_checkpoint(f"Сгенерированы новые запросы: {new_queries}")
                else:
                    resource_manager.log_checkpoint("Refinement-агент не сгенерировал новых запросов, прерываем цикл")
                    break
            else:
                 resource_manager.log_checkpoint("Нет инф. пробелов или достигнут лимит итераций, выходим из цикла")
                 break

        # --- Финальная Генерация ---
        resource_manager.log_checkpoint("Начало финальной генерации ответа")
        if not collected_docs:
            return "К сожалению, по вашему запросу не удалось найти релевантную информацию в базе знаний."

        # --- ИЗМЕНЕНИЕ: Используем тот же обрезанный контекст для финального ответа ---
        final_docs_for_generation = collected_docs[:self.config.MAX_CONTEXT_DOCS]
        final_context = "\n\n".join([doc.page_content for doc in final_docs_for_generation])
        
        final_prompt = PROMPT_LIBRARY["final_generator"].format(
            context=final_context,
            question=question
        )

        answer = self.llm_client.generate(prompt=final_prompt, model_name=self.config.GENERATOR_MODEL)
        
        resource_manager.log_checkpoint(f"Завершение обработки вопроса")
        return answer