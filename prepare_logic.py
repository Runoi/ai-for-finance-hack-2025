"""
Модуль для выполнения всей "офлайн" работы по созданию поисковых индексов.

Этот скрипт содержит единую функцию `prepare_all_indices`, которая:
1. Читает исходные данные из `train_data.csv`.
2. Разбивает текст на семантические чанки с помощью `MarkdownHeaderTextSplitter`.
3. Создает и сохраняет 4 поисковых артефакта для гибридного поиска:
    - Голова A: Семантический индекс FAISS по тексту.
    - Голова B: Индекс по ключевым словам TFIDFRetriever.
    - Голова C: Семантический индекс FAISS по метаданным.
    - Голова D: Граф Концептов на базе NetworkX.
"""

import os
import pandas as pd
import networkx as nx
import pickle
from typing import List, Dict

# --- Импорты LangChain---
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.embeddings import Embeddings
# --- Импорты scikit-learn ---
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# --- Импорты наших модулей ---
from config import config
from utils.resource_manager import resource_manager
from rag_components.llm_client import LLMClient

from rag_components.llm_client import LLMClient
from rag_components.embeddings import LiteLLMEmbeddings


def prepare_all_indices():
    """
    Основная функция, запускающая полный процесс создания всех поисковых артефактов.
    Проверяет наличие данных и запускает каждый этап, логируя прогресс.
    """
    resource_manager.log_checkpoint("Начало полной подготовки индексов")

    try:
        df = pd.read_csv(config.TRAIN_DATA_PATH)
        df = df.dropna(subset=['text'])
        df['doc_id'] = df['id']
    except (FileNotFoundError, KeyError) as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось прочитать {config.TRAIN_DATA_PATH}. Ошибка: {e}")
        return

    # --- 1. Общий этап: Чанкинг ---
    resource_manager.log_checkpoint("Этап 1: Разбиение на чанки")
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    all_docs: List[Document] = []
    for index, row in df.iterrows():
        chunks = text_splitter.split_text(row['text'])
        for chunk in chunks:
            chunk.metadata.update({
                'doc_id': row.get('id', 'unknown'),
                'annotation': row.get('annotation', ''),
                'tags': str(row.get('tags', []))
            })
        all_docs.extend(chunks)
    resource_manager.log_checkpoint(f"Данные разбиты на {len(all_docs)} чанков")

    # --- Инициализация клиента для эмбеддингов с явным assert ---
    llm_client = LLMClient()
    embed_api_key = llm_client.embed_api_key
    
    if embed_api_key:
        embedding_function = LiteLLMEmbeddings(client=llm_client)

        # --- 2. Создание Артефактов для каждой "Головы" ---
        if not os.path.exists(config.STORAGE_PATH):
            os.makedirs(config.STORAGE_PATH)

        _create_text_faiss_index(all_docs, embedding_function)
        _create_tfidf_retriever(all_docs)
        _create_metadata_faiss_index(df, embedding_function)
        _create_concept_graph(all_docs)
        
        resource_manager.log_checkpoint("Все индексы созданы и сохранены")
    else:
        print("!!! КРИТИЧЕСКАЯ ОШИБКА: API-ключ для эмбеддингов не найден. Процесс создания индексов прерван.")
        return
    # --- 2. Создание Артефактов для каждой "Головы" ---
    if not os.path.exists(config.STORAGE_PATH):
        os.makedirs(config.STORAGE_PATH)

    _create_text_faiss_index(all_docs, embedding_function)
    _create_tfidf_retriever(all_docs)
    _create_metadata_faiss_index(df, embedding_function)
    _create_concept_graph(all_docs)
    with open(os.path.join(config.STORAGE_PATH, "all_docs.pkl"), "wb") as f:
        pickle.dump(all_docs, f)
    resource_manager.log_checkpoint("-> Список all_docs сохранен")
    
    resource_manager.log_checkpoint("Все индексы созданы и сохранены")


def _create_text_faiss_index(docs: List[Document], embeddings: Embeddings):
    """
    Создает и сохраняет FAISS индекс по основному тексту чанков,
    используя пакетную обработку для обхода лимитов API.
    """
    resource_manager.log_checkpoint("-> Голова A: Создание FAISS (текст)")
    
    batch_size = 256  # Оптимальный размер батча, можно тюнить
    vectorstore = None

    for i in tqdm(range(0, len(docs), batch_size), desc="Создание FAISS батчами"):
        batch = docs[i:i + batch_size]
        if not batch:
            continue
            
        if vectorstore is None:
            # Создаем индекс на первом батче
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            # Добавляем последующие батчи в существующий индекс
            vectorstore.add_documents(batch)
            
    if vectorstore:
        vectorstore.save_local(os.path.join(config.STORAGE_PATH, "faiss_text_index"))
        resource_manager.log_checkpoint("-> Голова A: Индекс сохранен")
    else:
        print("!!! WARNING: Не удалось создать FAISS индекс (документы отсутствуют).")


def _create_tfidf_retriever(docs: List[Document]):
    """Создает и сохраняет TfidfRetriever по тексту чанков."""
    # (Эта функция не использует embeddings, поэтому без изменений)
    resource_manager.log_checkpoint("-> Голова B: Создание TF-IDF ретривера")
    tfidf_retriever = TFIDFRetriever.from_documents(docs)
    with open(os.path.join(config.STORAGE_PATH, "tfidf_retriever.pkl"), "wb") as f:
        pickle.dump(tfidf_retriever, f)
    resource_manager.log_checkpoint("-> Голова B: Ретривер сохранен")


def _create_metadata_faiss_index(df: pd.DataFrame, embeddings: Embeddings):
    """
    Создает и сохраняет FAISS индекс по метаданным.
    Здесь данных меньше, батчинг может не понадобиться, но добавим для надежности.
    """
    resource_manager.log_checkpoint("-> Голова C: Создание FAISS (мета)")
    meta_docs: List[Document] = []
    for index, row in df.iterrows():
        meta_content = f"Аннотация: {row.get('annotation', '')}\nТеги: {str(row.get('tags', []))}"
        meta_doc = Document(page_content=meta_content, metadata={'doc_id': row.get('id', 'unknown')})
        meta_docs.append(meta_doc)

    # Здесь тоже используем from_documents, так как он эффективен для списков
    # и количество мета-документов (350) не должно превысить лимит.
    # Если бы превысило, мы бы применили ту же логику с батчингом.
    if meta_docs:
        vectorstore = FAISS.from_documents(meta_docs, embeddings)
        vectorstore.save_local(os.path.join(config.STORAGE_PATH, "faiss_meta_index"))
        resource_manager.log_checkpoint("-> Голова C: Индекс сохранен")
    else:
        print("!!! WARNING: Не удалось создать FAISS (мета) индекс.")


def _create_concept_graph(docs: List[Document]):
    """Извлекает концепты и строит граф их совместной встречаемости."""
    resource_manager.log_checkpoint("-> Голова D: Создание Графа Концептов")
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1,2))
    doc_texts = [doc.page_content for doc in docs]
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    G = nx.Graph()
    doc_to_concepts: Dict[str, List[str]] = {}

    for i, doc in enumerate(docs):
        # Доступ к строке разреженной матрицы. Pylance может ругаться из-за неполных
        # type stubs для scipy, но этот код корректен. Подавляем ложное срабатывание.
        row_slice = tfidf_matrix[i]  # type: ignore
        feature_indices = row_slice.indices
        
        try:
            doc_concepts = [feature_names[j] for j in feature_indices]
        except IndexError:
            doc_concepts = []
            
        doc_id = doc.metadata.get('doc_id', f'chunk_{i}')
        if doc_id not in doc_to_concepts:
            doc_to_concepts[doc_id] = []
        doc_to_concepts[doc_id].extend(doc_concepts)

    for doc_id, concepts in doc_to_concepts.items():
        unique_concepts = list(set(concepts))
        for i in range(len(unique_concepts)):
            for j in range(i + 1, len(unique_concepts)):
                c1, c2 = unique_concepts[i], unique_concepts[j]
                if G.has_edge(c1, c2):
                    G[c1][c2]['weight'] += 1
                else:
                    G.add_edge(c1, c2, weight=1)
    
    with open(os.path.join(config.STORAGE_PATH, "concept_graph.gpickle"), "wb") as f:
        pickle.dump(G, f)
    with open(os.path.join(config.STORAGE_PATH, "doc_to_concepts.pkl"), "wb") as f:
        pickle.dump(doc_to_concepts, f)
        
    resource_manager.log_checkpoint("-> Голова D: Граф сохранен")