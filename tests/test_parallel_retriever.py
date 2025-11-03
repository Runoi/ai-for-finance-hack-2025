import sys
import os
import time
from dotenv import load_dotenv

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
from rag_components.retrievers import build_ensemble_retriever
from config import config

TEST_QUESTION = "Чем отличаются ОСАГО, каско, ДСАГО по выплатам?"

def run_retriever_test():
    load_dotenv()
    
    # --- Синхронный Тест ---
    print("\n" + "="*80)
    print("ТЕСТ 1: СИНХРОННЫЙ РЕТРИВЕР")
    print("="*80)
    
    # Временно отключаем параллелизм в конфиге
    config.ENABLE_PARALLEL_REQUESTS = False 
    retriever_sync = build_ensemble_retriever()
    
    start_time_sync = time.time()
    docs_sync = retriever_sync.invoke(TEST_QUESTION)
    end_time_sync = time.time()
    sync_duration = end_time_sync - start_time_sync
    
    print(f"Найдено документов: {len(docs_sync)}")
    print(f"--- Результат: {sync_duration:.2f} секунд ---")

    # --- Параллельный Тест ---
    print("\n" + "="*80)
    print("ТЕСТ 2: ПАРАЛЛЕЛЬНЫЙ РЕТРИВЕР")
    print("="*80)

    # Включаем параллелизм
    config.ENABLE_PARALLEL_REQUESTS = True
    retriever_parallel = build_ensemble_retriever()

    start_time_parallel = time.time()
    docs_parallel = retriever_parallel.invoke(TEST_QUESTION)
    end_time_parallel = time.time()
    parallel_duration = end_time_parallel - start_time_parallel

    print(f"Найдено документов: {len(docs_parallel)}")
    print(f"--- Результат: {parallel_duration:.2f} секунд ---")

    # --- Финальный Отчет ---
    print("\n\n" + "="*30 + " СРАВНИТЕЛЬНЫЙ ОТЧЕТ " + "="*30)
    print(f"Синхронный поиск:    {sync_duration:.2f} с")
    print(f"Параллельный поиск: {parallel_duration:.2f} с")
    if parallel_duration < sync_duration:
        speedup = (sync_duration / parallel_duration)
        print(f"Ускорение поиска: x{speedup:.2f}")
    print("="*80)

if __name__ == "__main__":
    run_retriever_test()

# ============================== СРАВНИТЕЛЬНЫЙ ОТЧЕТ ==============================
# Синхронный поиск:    2.14 с
# Параллельный поиск: 1.32 с
# Ускорение поиска: x1.63
# ================================================================================