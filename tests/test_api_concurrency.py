"""
Скрипт для стресс-тестирования API и определения оптимального
уровня параллелизма.
"""

import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# --- Настройка Окружения ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Импорты ---
from rag_components.llm_client import LLMClient
from config import config

# --- Конфигурация Теста ---
# Простой, короткий вопрос для теста
TEST_PROMPT = "Напиши 'привет'"
# Модель для теста (быстрая, чтобы измерять именно сетевую задержку, а не генерацию)
TEST_MODEL = config.GENERATOR_MODEL 
# Количество запросов, которое мы отправим для каждого уровня параллелизма
NUM_REQUESTS_PER_RUN = 20
# Уровни параллелизма, которые мы хотим протестировать
WORKER_LEVELS = [1, 2, 4, 8, 12, 16]

# Глобальная переменная для подсчета ошибок
error_count = 0

def make_single_request(client: LLMClient):
    """
    Выполняет один API-запрос и обрабатывает возможные ошибки.
    """
    global error_count
    try:
        client.generate(prompt=TEST_PROMPT, model_name=TEST_MODEL)
    except Exception as e:
        print(f"\n!!! Ошибка в потоке: {e}")
        with threading.Lock():
            error_count += 1

def run_concurrency_test():
    load_dotenv()
    
    print("Инициализация LLM-клиента...")
    llm_client = LLMClient()
    
    results = {}

    for num_workers in WORKER_LEVELS:
        print("\n" + "="*80)
        print(f"ТЕСТ: {num_workers} параллельных воркеров")
        print("="*80)
        
        # Сбрасываем счетчик ошибок перед каждым запуском
        global error_count
        error_count = 0
        
        # Создаем список задач
        tasks = [llm_client] * NUM_REQUESTS_PER_RUN
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Запускаем NUM_REQUESTS_PER_RUN запросов, используя `num_workers` потоков
            list(executor.map(make_single_request, tasks))
            
        end_time = time.time()
        
        duration = end_time - start_time
        # Запросы в секунду (RPS)
        requests_per_second = NUM_REQUESTS_PER_RUN / duration if duration > 0 else float('inf')
        
        results[num_workers] = {
            "duration": duration,
            "rps": requests_per_second,
            "errors": error_count
        }

        print(f"--- Результат для {num_workers} воркеров ---")
        print(f"  Общее время: {duration:.2f} с")
        print(f"  Запросов в секунду (RPS): {requests_per_second:.2f}")
        print(f"  Количество ошибок: {error_count}")
    
    # --- Финальный Сводный Отчет ---
    print("\n\n" + "="*30 + " СВОДНЫЙ ОТЧЕТ ПО ПАРАЛЛЕЛИЗМУ " + "="*30)
    print(f"{'Воркеры':<10} | {'Время (с)':<15} | {'RPS':<10} | {'Ошибки':<10}")
    print("-" * 55)
    for workers, res in results.items():
        print(f"{workers:<10} | {res['duration']:<15.2f} | {res['rps']:<10.2f} | {res['errors']:<10}")
    print("="*80)
    
    # Ищем оптимальное значение
    best_config = max(
        (w for w, r in results.items() if r['errors'] == 0), 
        key=lambda w: results[w]['rps'],
        default=None
    )
    if best_config:
        print(f"\n Оптимальное количество воркеров (макс. RPS без ошибок): {best_config}")
    else:
        print("\n Не удалось найти конфигурацию без ошибок.")


if __name__ == "__main__":
    run_concurrency_test()