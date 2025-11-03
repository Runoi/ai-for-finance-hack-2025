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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
# --- Импорты LangChain ---
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TFIDFRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.embeddings import Embeddings
# --- Импорты наших модулей ---
from config import config
from utils.resource_manager import resource_manager
from rag_components.embeddings import LiteLLMEmbeddings
from rag_components.llm_client import LLMClient


class ConceptGraphRetriever(BaseRetriever):
    """
    Кастомный ретривер, использующий иерархический поиск по графу концептов.

    Поиск происходит в два этапа:
    1.  Определение наиболее релевантных "тем" (кластеров концептов).
    2.  Поиск (обход графа) внутри этих тем для нахождения связанных документов.
    """
    graph: nx.Graph
    doc_to_concepts: Dict[str, List[str]]
    all_docs: List[Document]
    concept_to_cluster: Dict[str, str]
    embedding_function: Embeddings
    
    all_docs_map: Dict[str, Document] = {}

    def __init__(self, **data: Any):
        """Инициализирует ретривер и создает карту документов для быстрого доступа."""
        super().__init__(**data)
        # Создаем карту doc_id -> Document для O(1) доступа
        self.all_docs_map = {
            doc.metadata.get('doc_id', f'fallback_id_{i}'): doc 
            for i, doc in enumerate(self.all_docs)
        }
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Выполняет иерархический поиск: сначала по темам, затем по концептам.
        """
        # --- Шаг 1: Определение релевантных тем ---
        
        # Извлекаем "сырые" концепты из текста запроса.
        query_concepts = set(query.lower().split())
        
        # Находим те из них, что присутствуют в нашем графе.
        start_nodes = [node for node in query_concepts if node in self.graph]
        if not start_nodes:
            return []

        # "Голосуем" за кластеры. Каждый концепт из запроса добавляет "голос" своему кластеру.
        cluster_scores = defaultdict(int)
        for concept in start_nodes:
            if concept in self.concept_to_cluster:
                cluster_id = self.concept_to_cluster[concept]
                cluster_scores[cluster_id] += 1
        
        if not cluster_scores:
            # Если ни один концепт не попал в кластер, отступаем к простому поиску
            top_clusters = None 
        else:
            # Выбираем топ-5 самых релевантных кластеров.
            top_clusters = {
                cid for cid, score in sorted(
                    cluster_scores.items(), key=lambda item: item[1], reverse=True
                )[:5]
            }

        # --- Шаг 2: Обход графа внутри релевантных тем ---
        
        related_concepts = set(start_nodes)
        for node in start_nodes:
            try:
                for neighbor in self.graph.neighbors(node):
                    # Если у нас есть топ-кластеры, добавляем соседа, только если
                    # он принадлежит к одному из них.
                    if top_clusters is None or self.concept_to_cluster.get(neighbor) in top_clusters:
                        related_concepts.add(neighbor)
            except nx.NetworkXError:
                # Если у узла нет соседей, просто пропускаем его.
                continue
        
        # --- Шаг 3: Поиск документов по финальному набору концептов ---
        
        relevant_doc_ids = set()
        for doc_id, concepts in self.doc_to_concepts.items():
            # Проверяем, есть ли пересечение между концептами документа и нашим расширенным набором.
            if concepts and not related_concepts.isdisjoint(concepts):
                relevant_doc_ids.add(doc_id)
        
        relevant_docs = [
            self.all_docs_map[doc_id] for doc_id in relevant_doc_ids if doc_id in self.all_docs_map
        ]
        
        return relevant_docs

class ParallelEnsembleRetriever(EnsembleRetriever):
    """
    Расширенная версия EnsembleRetriever, которая выполняет запросы
    к дочерним ретриверам параллельно с использованием ThreadPoolExecutor.
    """
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Параллельно получает документы от всех ретриверов и объединяет их.
        """
        # --- ПАРАЛЛЕЛЬНЫЙ ВЫЗОВ ---
        with ThreadPoolExecutor(max_workers=len(self.retrievers)) as executor:
            # .map применяет invoke к каждому ретриверу в отдельном потоке
            retriever_docs = list(executor.map(
                lambda r: r.invoke(query, config={"callbacks": run_manager.get_child()}),
                self.retrievers
            ))

        # Дальнейшая логика - это Reciprocal Rank Fusion,
        # скопированная из исходников LangChain.
        fused_scores: Dict[str, float] = defaultdict(float)
        for docs_list, weight in zip(retriever_docs, self.weights):
            for rank, doc in enumerate(docs_list):
                # Используем page_content как уникальный ключ документа
                if doc.page_content not in fused_scores:
                    fused_scores[doc.page_content] = 0
                fused_scores[doc.page_content] += weight / (self.c + rank + 1)
        
        # Собираем все уникальные документы в один список
        all_unique_docs_map = {doc.page_content: doc for docs_list in retriever_docs for doc in docs_list}
        
        # Сортируем документы по их RRF-оценке
        sorted_contents = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        final_docs = [all_unique_docs_map[content] for content in sorted_contents]
        
        return final_docs

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
    
    
    # --- Голова A: Семантический поиск по тексту (с MMR) ---
    if config.ENABLE_SEMANTIC_HEAD:
        try:
            path = os.path.join(config.STORAGE_PATH, "faiss_text_index")
            vectorstore = FAISS.load_local(path, embedding_function, allow_dangerous_deserialization=True)
            
            
            retriever_a = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': config.MMR_K, 
                    'fetch_k': config.MMR_FETCH_K,
                    'lambda_mult': config.MMR_LAMBDA_MULT
                }
            )
            retriever_list.append(retriever_a)
            weights_list.append(config.RETRIEVER_WEIGHTS[0])
            resource_manager.log_checkpoint("-> Голова A (FAISS Текст с MMR) загружена")
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

    # --- Голова C: Семантический поиск по метаданным (тоже с MMR) ---
    if config.ENABLE_METADATA_HEAD:
        try:
            path = os.path.join(config.STORAGE_PATH, "faiss_meta_index")
            vectorstore_meta = FAISS.load_local(path, embedding_function, allow_dangerous_deserialization=True)

            
            # Для мета-поиска можно использовать другие параметры k
            retriever_c = vectorstore_meta.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': 10, 
                    'fetch_k': 30,
                    'lambda_mult': config.MMR_LAMBDA_MULT
                }
            )
            retriever_list.append(retriever_c)
            weights_list.append(config.RETRIEVER_WEIGHTS[2])
            resource_manager.log_checkpoint("-> Голова C (FAISS Мета с MMR) загружена")
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
            with open(os.path.join(config.STORAGE_PATH, "concept_to_cluster.pkl"), "rb") as f:
                concept_to_cluster = pickle.load(f)
            
                
            graph_retriever = ConceptGraphRetriever(
                graph=graph, 
                doc_to_concepts=doc_to_concepts, 
                all_docs=all_docs,
                concept_to_cluster=concept_to_cluster,
                embedding_function=embedding_function
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
        
        return retriever_list[0] # type: ignore

    # --- ЛОГИКА ПЕРЕКЛЮЧЕНИЯ ---
    if config.ENABLE_PARALLEL_REQUESTS:
        ensemble_retriever = ParallelEnsembleRetriever(
            retrievers=retriever_list, weights=weights_list
        )
        resource_manager.log_checkpoint("ParallelEnsembleRetriever успешно собран")
    else:
        ensemble_retriever = EnsembleRetriever(
            retrievers=retriever_list, weights=weights_list
        )
        resource_manager.log_checkpoint("EnsembleRetriever (classic) успешно собран")
    
    resource_manager.log_checkpoint("EnsembleRetriever (classic) успешно собран")
    return ensemble_retriever