
import sys
import os
import time
from dotenv import load_dotenv
from typing import Optional

# --- УНИВЕРСАЛЬНЫЙ БЛОК ДЛЯ ИСПРАВЛЕНИЯ ИМПОРТОВ ---
# Добавляем корень проекта в sys.path, чтобы импорты работали из любой папки.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Идем вверх по дереву, пока не найдем корневой маркер (например, requirements.txt)
project_root = current_dir
while not os.path.exists(os.path.join(project_root, 'requirements.txt')):
    project_root = os.path.dirname(project_root)
    if project_root == os.path.dirname(project_root):
        raise FileNotFoundError("Не удалось найти корень проекта.")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- КОНЕЦ БЛОКА ---

# --- Импорты ---
from config import Config
from rag_components.llm_client import LLMClient
from rag_components.agents import DecompositionAgent
from rag_components.retrievers import build_ensemble_retriever

from rag_components.pipeline import RAGPipeline 

TEST_QUESTION = "Чем отличаются госгарантии по деньгам на эскроу от гарантий по накоплениям в НПФ?"

def run_performance_test():
    load_dotenv()
    
    # --- Подготовка (один раз) ---
    print("Загрузка общих компонентов...")
    llm_client = LLMClient()
    ensemble_retriever = build_ensemble_retriever()
    decomposition_agent = DecompositionAgent(llm_client=llm_client)

    # --- Тест 1: Синхронный Режим ---
    print("\n" + "="*80)
    print("ТЕСТ 1: СИНХРОННЫЙ РЕЖИМ (ENABLE_PARALLEL_REQUESTS = False)")
    print("="*80)
    
    sync_config = Config()
    sync_config.STRATEGY = 'DECOMPOSE'
    sync_config.ENABLE_PARALLEL_REQUESTS = False

    pipeline_sync = RAGPipeline(
        retriever=ensemble_retriever,
        llm_client=llm_client,
        decomposition_agent=decomposition_agent,
        # Остальные агенты не используются в этой стратегии, передаем заглушки
        sea_agent=None, refinement_agent=None, # type: ignore
        config_override=sync_config
    )

    start_time_sync = time.time()
    # Выполняем только декомпозицию и поиск
    sub_queries = pipeline_sync.decomposition_agent.decompose(TEST_QUESTION)
    for q in sub_queries:
        pipeline_sync.retriever.invoke(q)
    end_time_sync = time.time()
    sync_duration = end_time_sync - start_time_sync
    print(f"--- Результат: {sync_duration:.2f} секунд ---")

    # --- Тест 2: Параллельный Режим ---
    print("\n" + "="*80)
    print("ТЕСТ 2: ПАРАЛЛЕЛЬНЫЙ РЕЖИМ (ENABLE_PARALLEL_REQUESTS = True)")
    print("="*80)

    parallel_config = Config()
    parallel_config.STRATEGY = 'DECOMPOSE'
    parallel_config.ENABLE_PARALLEL_REQUESTS = True
    
    pipeline_parallel = RAGPipeline(
        retriever=ensemble_retriever,
        llm_client=llm_client,
        decomposition_agent=decomposition_agent,
        sea_agent=None, refinement_agent=None, # type: ignore
        config_override=parallel_config
    )

    start_time_parallel = time.time()
    # Выполняем только декомпозицию и поиск
    sub_queries_p = pipeline_parallel.decomposition_agent.decompose(TEST_QUESTION)
    with ThreadPoolExecutor(max_workers=len(sub_queries_p)) as executor:
        list(executor.map(pipeline_parallel.retriever.invoke, sub_queries_p))
    end_time_parallel = time.time()
    parallel_duration = end_time_parallel - start_time_parallel
    print(f"--- Результат: {parallel_duration:.2f} секунд ---")

    # --- Финальный Отчет ---
    print("\n\n" + "="*30 + " СРАВНИТЕЛЬНЫЙ ОТЧЕТ " + "="*30)
    print(f"Синхронное выполнение: {sync_duration:.2f} с")
    print(f"Параллельное выполнение: {parallel_duration:.2f} с")
    if parallel_duration < sync_duration:
        speedup = (sync_duration / parallel_duration)
        print(f"Ускорение: x{speedup:.2f}")
    print("="*80)


if __name__ == "__main__":
    # Импортируем ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    run_performance_test()


# ============================== СРАВНИТЕЛЬНЫЙ ОТЧЕТ ==============================
# Синхронное выполнение: 7.20 с
# Параллельное выполнение: 3.16 с
# Ускорение: x2.28
# ================================================================================