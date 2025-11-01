"""
Интерактивный скрипт для отладки RAG-пайплайна на одном вопросе.

Этот скрипт загружает полную архитектуру и позволяет задавать вопросы
в командной строке, чтобы тестировать и анализировать поведение системы
в реальном времени.
"""

import sys
import os
from dotenv import load_dotenv

# --- Настройка Окружения (та же, что и в main.py) ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Импорты Наших Модулей ---
from config import config
from utils.resource_manager import ResourceManager # Импортируем класс, а не синглтон
from rag_components.llm_client import LLMClient
from rag_components.agents import SeaAgent, RefinementAgent
from rag_components.retrievers import build_ensemble_retriever
from rag_components.pipeline import RAGPipeline
from prepare_logic import prepare_all_indices


def main():
    """
    Основная функция для интерактивной сессии.
    """
    # Загружаем API-ключи
    load_dotenv()
    
    # Создаем новый, "чистый" ResourceManager для этой сессии
    session_rm = ResourceManager()

    # --- Сборка Пайплайна (аналогично main.py) ---
    session_rm.log_checkpoint("Старт сессии отладки")

    if not os.path.exists(os.path.join(config.STORAGE_PATH, "index.faiss")):
        print("Индексы не найдены. Запускаю процесс создания...")
        prepare_all_indices()
    
    try:
        session_rm.log_checkpoint("Сборка RAG-пайплайна...")
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
        session_rm.log_checkpoint("RAG-пайплайн успешно собран")
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

            # Сбрасываем счетчики для одного вопроса
            # (Если мы хотим считать общие затраты, эту часть можно убрать)
            session_rm = ResourceManager() 

            # --- ЗАПУСК ОБРАБОТКИ ОДНОГО ВОПРОСА ---
            answer = pipeline.run(question=question)
            
            print("\n" + "-"*15 + " [ОТВЕТ МОДЕЛИ] " + "-"*15)
            print(answer)
            print("-" * (32 + 2))
            
            # --- Отчет по ресурсам для этого вопроса ---
            print("\n--- Отчет по ресурсам для этого запроса ---")
            summary = session_rm.get_summary()
            for key, value in summary.items():
                if "usd" in key:
                    print(f"{key:<20}: ${value:.5f}")
                elif "sec" in key:
                    print(f"{key:<20}: {value:.2f} s")
                else:
                    print(f"{key:<20}: {value:.2f} MB")
            print("-" * 45)

        except KeyboardInterrupt:
            print("\nВыход из программы.")
            break
        except Exception as e:
            print(f"\n!!! Произошла ошибка во время выполнения: {e}")

if __name__ == "__main__":
    main()