"""
Скрипт для сравнительного тестирования трех RAG-стратегий на одном вопросе.

Запускает пайплайн в режимах 'SIMPLE', 'DECOMPOSE' и 'ITERATIVE'
и выводит сравнительный отчет по качеству ответа, времени и стоимости.
"""

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


# --- Импорты Наших Модулей ---
from config import Config
from utils.resource_manager import ResourceManager
from rag_components.llm_client import LLMClient
from rag_components.agents import SeaAgent, RefinementAgent, DecompositionAgent
from rag_components.retrievers import build_ensemble_retriever
from rag_components.pipeline import RAGPipeline
from prepare_logic import prepare_all_indices

# --- Тестовые Данные ---
TEST_QUESTION = "Чем отличаются госгарантии по деньгам на эскроу от гарантий по накоплениям в НПФ?"

def run_strategy_test():
    """
    Основная функция, запускающая тест для всех стратегий.
    """
    load_dotenv()
    
    # --- Подготовка (выполняется один раз) ---
    if not os.path.exists(os.path.join(Config.STORAGE_PATH, "all_docs.pkl")):
        print("Индексы не найдены. Запускаю процесс создания...")
        prepare_all_indices()
    
    print("Загрузка общих компонентов...")
    llm_client = LLMClient()
    ensemble_retriever = build_ensemble_retriever()
    sea_agent = SeaAgent(llm_client=llm_client)
    refinement_agent = RefinementAgent(llm_client=llm_client)
    decomposition_agent = DecompositionAgent(llm_client=llm_client)
    
    strategies_to_test = ['SIMPLE', 'DECOMPOSE', 'ITERATIVE']
    results = {}

    for strategy in strategies_to_test:
        print("\n" + "="*80)
        print(f"ЗАПУСК ТЕСТА ДЛЯ СТРАТЕГИИ: '{strategy}'")
        print("="*80)
        
        # Создаем временный конфиг для этого запуска
        test_config = Config()
        test_config.STRATEGY = strategy
        # Для итеративной стратегии оставим 2 итерации, чтобы увидеть разницу
        if strategy == 'ITERATIVE':
            test_config.MAX_ITERATIONS = 2  # type: ignore
        else:
            test_config.MAX_ITERATIONS = 1

        # Собираем пайплайн с временным конфигом
        pipeline = RAGPipeline(
            retriever=ensemble_retriever,
            llm_client=llm_client,
            sea_agent=sea_agent,
            refinement_agent=refinement_agent,
            decomposition_agent=decomposition_agent,
            config_override=test_config
        )

        # Используем отдельный ResourceManager для чистоты замеров
        rm = ResourceManager()
        
        start_time = time.time()
        answer = pipeline.run(question=TEST_QUESTION)
        end_time = time.time()
        
        run_time = end_time - start_time
        summary = rm.get_summary()

        print("\n" + "-"*15 + f" [ОТВЕТ ({strategy})] " + "-"*15)
        print(answer)
        print("-" * 50)
        
        results[strategy] = {
            "time": run_time,
            "cost": summary['api_spent_usd'],
            "answer": answer
        }
    
    # --- Вывод Финального Сводного Отчета ---
    print("\n\n" + "="*30 + " СРАВНИТЕЛЬНЫЙ ОТЧЕТ " + "="*30)
    for strategy, res in results.items():
        print(f"\n--- Стратегия: {strategy} ---")
        print(f"  Время выполнения: {res['time']:.2f} с")
        print(f"  Стоимость API:    ${res['cost']:.8f}")
    print("="*80)


if __name__ == "__main__":
    run_strategy_test()

#Синхроное/1 поток
#Сравнительный Анализ Трех Стратегий
# 1. SIMPLE (Поиск -> Генерация)
# Время: 18.05 с (очень быстро).
# Стоимость: $0.00041.
# Качество Ответа: Хорошее. Ответ правильный, структурированный, содержит ключевые отличия. Однако он менее подробный, чем у других стратегий. Например, он не упоминает, что лимит по эскроу после сделки снижается до 1.4 млн.
# Вердикт: Отличный, быстрый baseline. Подходит для простых вопросов.
# 2. DECOMPOSE (Декомпозиция -> Поиск -> Генерация)
# Время: 27.88 с (+~10 секунд к SIMPLE).
# Стоимость: $0.00098.
# Качество Ответа: Отличное. Ответ более подробный и лучше структурирован, чем у SIMPLE. Он правильно выделяет ключевые аспекты (объект, сроки, управление) и содержит важную деталь про снижение лимита по эскроу до 1.4 млн.
# Вердикт: Явный победитель по соотношению "цена/качество" для сравнительных вопросов. Десять дополнительных секунд на декомпозицию полностью окупаются глубиной ответа.
# 3. ITERATIVE (Поиск -> SEA -> Refine -> Поиск -> SEA -> Генерация)
# Время: 49.23 с (самый медленный, в ~2.7 раза медленнее SIMPLE).
# Стоимость: $0.00464 (самый дорогой).
# Качество Ответа: Превосходное. Это самый подробный, точный и хорошо структурированный ответ из всех. Он включает все детали из ответа DECOMPOSE и добавляет новые (про "инвестиционные убытки" в НПФ, про СФР). SeaAgent дважды правильно решил, что информации недостаточно, что заставило систему "копать глубже".
# Вердикт: Демонстрирует наивысшее качество, но ценой значительного увеличения времени и стоимости.

#Параллелизм Декомпоз
#============================== СРАВНИТЕЛЬНЫЙ ОТЧЕТ ==============================

# --- Стратегия: SIMPLE ---
#   Время выполнения: 16.42 с
#   Стоимость API:    $0.00000000

# --- Стратегия: DECOMPOSE ---
#   Время выполнения: 19.28 с
#   Стоимость API:    $0.00000000

# --- Стратегия: ITERATIVE ---
#   Время выполнения: 49.20 с
#   Стоимость API:    $0.00000000
# ================================================================================