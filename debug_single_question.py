"""
Интерактивный скрипт для отладки RAG-пайплайна на одном вопросе.
"""

import sys
import os
import time # <-- Добавляем импорт time
from dotenv import load_dotenv

# --- Настройка Окружения ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Импорты Наших Модулей ---
from config import config
from utils.resource_manager import resource_manager # <-- Импортируем глобальный синглтон
from rag_components.llm_client import LLMClient
from rag_components.agents import SeaAgent, RefinementAgent
from rag_components.retrievers import build_ensemble_retriever
from rag_components.pipeline import RAGPipeline
from prepare_logic import prepare_all_indices


def main():
    """
    Основная функция для интерактивной сессии.
    """
    load_dotenv()
    
    # Используем глобальный ResourceManager для всей сессии
    resource_manager.log_checkpoint("Старт сессии отладки")

    # --- Подготовка и сборка пайплайна (ВНЕ цикла) ---
    if not os.path.exists(os.path.join(config.STORAGE_PATH, "faiss_text_index", "index.faiss")):
        print("Индексы не найдены. Запускаю процесс создания...")
        prepare_all_indices()
    
    try:
        resource_manager.log_checkpoint("Сборка RAG-пайплайна...")
        llm_client = LLMClient()
        sea_agent = SeaAgent(llm_client=llm_client)
        refinement_agent = RefinementAgent(llm_client=llm_client)
        ensemble_retriever = build_ensemble_retriever()

        pipeline = RAGPipeline(
            retriever=ensemble_retriever,
            llm_client=llm_client,
            sea_agent=sea_agent,
            refinement_agent=refinement_agent
        )
        resource_manager.log_checkpoint("RAG-пайплайн успешно собран")
    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при сборке пайплайна: {e}")
        return

    # --- Интерактивный Цикл ---
    print("\n" + "="*30)
    print("  ОТЛАДОЧНЫЙ СТЕНД RAG")
    print("="*30)
    print("Введите ваш вопрос или 'exit' для выхода.")

    while True:
        try:
            question = input("\n[Ваш вопрос] > ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
            if not question.strip():
                continue

            # Запоминаем состояние ресурсов ДО запроса
            start_time = time.time()
            start_cost = resource_manager.api_cost_usd

            # --- ЗАПУСК ОБРАБОТКИ ОДНОГО ВОПРОСА ---
            answer = pipeline.run(question=question)
            
            # --- ВЫВОД РЕЗУЛЬТАТА ---
            print("\n" + "-"*15 + " [ОТВЕТ МОДЕЛИ] " + "-"*15)
            print(answer)
            print("-" * (32 + 2))
            
            # --- ОТЧЕТ ПО РЕСУРСАМ ДЛЯ ЭТОГО ЗАПРОСА ---
            end_time = time.time()
            end_cost = resource_manager.api_cost_usd
            
            print("\n--- Отчет по ресурсам для этого запроса ---")
            print(f"{'Время выполнения':<20}: {end_time - start_time:.2f} s")
            print(f"{'Стоимость API':<20}: ${end_cost - start_cost:.8f}")
            print("-" * 45)

        except KeyboardInterrupt:
            print("\nВыход из программы.")
            break
        except Exception as e:
            print(f"\n!!! Произошла ошибка во время выполнения: {e}")

if __name__ == "__main__":
    main()