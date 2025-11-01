"""
Модуль для создания и сборки гибридного ансамблевого ретривера.

Основная функция `build_ensemble_retriever` загружает все предварительно
созданные поисковые артефакты и собирает их в единый `EnsembleRetriever`,
который использует несколько стратегий поиска ("голов") параллельно.
"""

import os
import pickle
import networkx as nx
from typing import List

# --- Импорты LangChain ---
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TFIDFRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

# --- Импорты наших модулей ---
from config import config
from utils.resource_manager import resource_manager
from rag_components.llm_client import LLMClient
from rag_components.embeddings import LiteLLMEmbeddings


class ConceptGraphRetriever(BaseRetriever):
    """
    Кастомный ретривер для поиска по графу концептов.

    Реализует логику поиска связанных документов путем обхода графа
    ключевых слов (концептов), извлеченных из корпуса.
    """
    graph: nx.Graph
    doc_to_concepts: dict
    all_docs: List[Document]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Основной метод, выполняющий поиск по графу.

        1. Извлекает концепты из запроса.
        2. Находит эти концепты в графе.
        3. Выполняет обход в ширину (BFS) для нахождения связанных концептов.
        4. Находит все документы, содержащие найденные концепты.
        5. Возвращает уникальный список этих документов.
        """
        # Простая токенизация запроса для извлечения стартовых концептов
        query_concepts = set(query.lower().split())
        
        start_nodes = [node for node in query_concepts if node in self.graph]
        if not start_nodes:
            return []

        # Обход графа на 1 уровень для расширения концептов
        related_concepts = set(start_nodes)
        for node in start_nodes:
            for neighbor in self.graph.neighbors(node):
                related_concepts.add(neighbor)
        
        # Находим все doc_id, связанные с найденными концептами
        relevant_doc_ids = set()
        for doc_id, concepts in self.doc_to_concepts.items():
            if not related_concepts.isdisjoint(concepts):
                relevant_doc_ids.add(doc_id)
        
        # Фильтруем исходные документы по найденным doc_id
        relevant_docs = [
            doc for doc in self.all_docs if doc.metadata.get('doc_id') in relevant_doc_ids
        ]
        return relevant_docs


def build_ensemble_retriever() -> BaseRetriever:
    """
    Собирает и возвращает настроенный EnsembleRetriever.

    Загружает все артефакты из папки `storage` и инициализирует
    каждую "голову" ретривера в соответствии с `config.py`.

    Returns:
        BaseRetriever: Готовый к использованию ансамблевый ретривер.
    """
    resource_manager.log_checkpoint("Начало сборки EnsembleRetriever")
    
    llm_client = LLMClient()
    embedding_function = LiteLLMEmbeddings(client=llm_client)
    
    retriever_list = []
    weights_list = []
    
    # --- Голова A: Семантический поиск по тексту ---
    if config.ENABLE_SEMANTIC_HEAD:
        try:
            path = os.path.join(config.STORAGE_PATH, "faiss_text_index")
            vectorstore = FAISS.load_local(path, embedding_function, allow_dangerous_deserialization=True)
            retriever_list.append(vectorstore.as_retriever())
            weights_list.append(config.RETRIEVER_WEIGHTS[0])
            resource_manager.log_checkpoint("-> Голова A (FAISS Текст) загружена")
        except Exception as e:
            print(f"!!! Ошибка загрузки FAISS (текст): {e}")

    # --- Голова B: Поиск по ключевым словам ---
    if config.ENABLE_KEYWORD_HEAD:
        try:
            path = os.path.join(config.STORAGE_PATH, "tfidf_retriever.pkl")
            with open(path, "rb") as f:
                tfidf_retriever = pickle.load(f)
            retriever_list.append(tfidf_retriever)
            weights_list.append(config.RETRIEVER_WEIGHTS[1])
            resource_manager.log_checkpoint("-> Голова B (TF-IDF) загружена")
        except Exception as e:
            print(f"!!! Ошибка загрузки TF-IDF: {e}")

    # --- Голова C: Семантический поиск по метаданным ---
    if config.ENABLE_METADATA_HEAD:
        try:
            path = os.path.join(config.STORAGE_PATH, "faiss_meta_index")
            vectorstore = FAISS.load_local(path, embedding_function, allow_dangerous_deserialization=True)
            retriever_list.append(vectorstore.as_retriever())
            weights_list.append(config.RETRIEVER_WEIGHTS[2])
            resource_manager.log_checkpoint("-> Голова C (FAISS Мета) загружена")
        except Exception as e:
            print(f"!!! Ошибка загрузки FAISS (мета): {e}")
            
    # --- Голова D: Поиск по Графу Концептов ---
    # ПРИМЕЧАНИЕ: Для этой головы нужно, чтобы all_docs был сохранен в prepare_logic.py
    # Мы добавим это в следующей итерации. Пока используем заглушку.
    if config.ENABLE_GRAPH_CONCEPT_HEAD:
        # TODO: Добавить сохранение и загрузку `all_docs.pkl`
        # Пока этот ретривер не будет работать без этого файла.
        print("!!! WARNING: Голова D (Граф) пока не реализована до конца (требует all_docs.pkl).")


    if not retriever_list:
        raise RuntimeError("Ни одна 'голова' ретривера не была успешно создана. Проверьте конфиг и наличие файлов в storage/")

    # Если только одна голова, возвращаем ее напрямую
    if len(retriever_list) == 1:
        resource_manager.log_checkpoint("Собран ретривер с одной головой")
        return retriever_list[0]

    # Собираем ансамбль
    ensemble_retriever = EnsembleRetriever(
        retrievers=retriever_list, weights=weights_list
    )
    
    resource_manager.log_checkpoint("EnsembleRetriever успешно собран")
    return ensemble_retriever