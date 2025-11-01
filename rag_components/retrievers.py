"""
Модуль для создания и сборки гибридного ансамблевого ретривера.

Основная функция `build_ensemble_retriever` загружает все предварительно
созданные поисковые артефакты и собирает их в единый `EnsembleRetriever`,
который использует несколько стратегий поиска ("голов") параллельно.
"""

import os
import pickle
import networkx as nx
from typing import List, Dict, Any

# --- Импорты LangChain ---
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TFIDFRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

# --- Импорты наших модулей ---
from config import config
from utils.resource_manager import resource_manager
from rag_components.embeddings import LiteLLMEmbeddings
from rag_components.llm_client import LLMClient


class ConceptGraphRetriever(BaseRetriever):
    """
    Кастомный ретривер для поиска по графу концептов.

    Реализует логику поиска связанных документов путем обхода графа
    ключевых слов (концептов), извлеченных из корпуса.
    """
    graph: nx.Graph
    doc_to_concepts: Dict[str, List[str]]
    all_docs: List[Document]
    all_docs_map: Dict[str, Document] = {}

    def __init__(self, **data: Any):
        """Инициализирует ретривер и создает карту документов для быстрого доступа."""
        super().__init__(**data)
        self.all_docs_map = {
            doc.metadata.get('doc_id', f'fallback_id_{i}'): doc 
            for i, doc in enumerate(self.all_docs)
        }
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Основной метод, выполняющий поиск по графу."""
        query_concepts = set(query.lower().split())
        start_nodes = [node for node in query_concepts if node in self.graph]
        if not start_nodes:
            return []

        related_concepts = set(start_nodes)
        for node in start_nodes:
            try:
                for neighbor in self.graph.neighbors(node):
                    related_concepts.add(neighbor)
            except nx.NetworkXError:
                continue
        
        relevant_doc_ids = set()
        for doc_id, concepts in self.doc_to_concepts.items():
            if concepts and not related_concepts.isdisjoint(concepts):
                relevant_doc_ids.add(doc_id)
        
        relevant_docs = [
            self.all_docs_map[doc_id] for doc_id in relevant_doc_ids if doc_id in self.all_docs_map
        ]
        return relevant_docs


def build_ensemble_retriever() -> BaseRetriever:
    """
    Собирает и возвращает настроенный EnsembleRetriever.
    """
    resource_manager.log_checkpoint("Начало сборки EnsembleRetriever")
    
    llm_client = LLMClient()
    embedding_function = LiteLLMEmbeddings(client=llm_client)
    
    # Явно указываем, что список будет содержать RetrieverLike объекты.
    retriever_list: List[RetrieverLike] = []
    weights_list: List[float] = []
    
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
                tfidf_retriever: TFIDFRetriever = pickle.load(f)
            retriever_list.append(tfidf_retriever)
            weights_list.append(config.RETRIEVER_WEIGHTS[1])
            resource_manager.log_checkpoint("-> Голова B (TF-IDF) загружена")
        except Exception as e:
            print(f"!!! Ошибка загрузки TF-IDF: {e}")

    # --- Голова C: Семантический поиск по метаданным ---
    if config.ENABLE_METADATA_HEAD:
        try:
            path = os.path.join(config.STORAGE_PATH, "faiss_meta_index")
            vectorstore_meta = FAISS.load_local(path, embedding_function, allow_dangerous_deserialization=True)
            retriever_list.append(vectorstore_meta.as_retriever())
            weights_list.append(config.RETRIEVER_WEIGHTS[2])
            resource_manager.log_checkpoint("-> Голова C (FAISS Мета) загружена")
        except Exception as e:
            print(f"!!! Ошибка загрузки FAISS (мета): {e}")
            
    # --- Голова D: Поиск по Графу Концептов ---
    if config.ENABLE_GRAPH_CONCEPT_HEAD:
        try:
            with open(os.path.join(config.STORAGE_PATH, "concept_graph.gpickle"), "rb") as f:
                graph = pickle.load(f)
            with open(os.path.join(config.STORAGE_PATH, "doc_to_concepts.pkl"), "rb") as f:
                doc_to_concepts = pickle.load(f)
            with open(os.path.join(config.STORAGE_PATH, "all_docs.pkl"), "rb") as f:
                all_docs: List[Document] = pickle.load(f)
                
            graph_retriever = ConceptGraphRetriever(
                graph=graph, 
                doc_to_concepts=doc_to_concepts, 
                all_docs=all_docs
            )
            retriever_list.append(graph_retriever)
            weights_list.append(config.RETRIEVER_WEIGHTS[3])
            resource_manager.log_checkpoint("-> Голова D (Граф) загружена")
        except Exception as e:
            print(f"!!! Ошибка загрузки Графа Концептов: {e}")


    if not retriever_list:
        raise RuntimeError("Ни одна 'голова' ретривера не была успешно создана.")

    if len(retriever_list) == 1:
        resource_manager.log_checkpoint("Собран ретривер с одной головой")
        # Мы знаем, что в списке один BaseRetriever, но для type checker-а это неизвестно
        # Используем type: ignore, чтобы подавить предупреждение
        return retriever_list[0] # type: ignore

    ensemble_retriever = EnsembleRetriever(
        retrievers=retriever_list, weights=weights_list
    )
    
    resource_manager.log_checkpoint("EnsembleRetriever (classic) успешно собран")
    return ensemble_retriever