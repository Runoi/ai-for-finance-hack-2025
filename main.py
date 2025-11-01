"""
Главный исполняемый файл проекта "Надежный Агент-Аналитик".

Этот скрипт является точкой входа и оркестратором всего процесса. Он выполняет
следующие шаги:
1. Настраивает пути для корректной работы импортов.
2. Загружает переменные окружения (API-ключи).
3. Проверяет наличие готовых поисковых индексов и запускает их создание
   при необходимости ("ленивая" инициализация).
4. Собирает полный RAG-пайплайн со всеми компонентами:
   - Гибридный ансамблевый ретривер.
   - LLM-клиент.
   - Агенты для итеративного анализа и уточнения.
5. Запускает обработку вопросов из `questions.csv`.
6. Сохраняет результаты в `submission.csv`.
7. Выводит финальный отчет о потраченных ресурсах.
"""

import sys
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# --- 1. Настройка Окружения ---
# Добавляем корневую папку проекта в sys.path для корректных абсолютных импортов.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. Импорты Наших Модулей (после настройки sys.path) ---
from config import config
from utils.resource_manager import resource_manager
from rag_components.llm_client import LLMClient
from rag_components.agents import SeaAgent, RefinementAgent
from rag_components.retrievers import build_ensemble_retriever
from rag_components.pipeline import RAGPipeline
from prepare_logic import prepare_all_indices


def main_workflow():
    """
    Основная функция, реализующая полный рабочий процесс.
    """
    resource_manager.log_checkpoint("Старт основного рабочего процесса")

    # --- 3. "Ленивая" Инициализация Индексов ---
    # Проверяем, существует ли хотя бы один ключевой файл индекса.
    # Если нет, запускаем полный, ресурсоемкий процесс подготовки.
    if not os.path.exists(os.path.join(config.STORAGE_PATH, "faiss_text_index", "index.faiss")):
        prepare_all_indices()
    else:
        resource_manager.log_checkpoint("Обнаружены готовые индексы. Пропускаем этап подготовки.")

    # --- 4. Сборка RAG-Пайплайна ---
    resource_manager.log_checkpoint("Сборка RAG-пайплайна...")
    try:
        # Инициализируем все компоненты
        llm_client = LLMClient()
        sea_agent = SeaAgent(llm_client=llm_client)
        refinement_agent = RefinementAgent(llm_client=llm_client)
        ensemble_retriever = build_ensemble_retriever()

        # Собираем главный пайплайн с помощью Dependency Injection
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

    # --- 5. Основной Цикл Обработки Вопросов ---
    try:
        questions_df = pd.read_csv(config.QUESTIONS_PATH)
    except FileNotFoundError:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Файл {config.QUESTIONS_PATH} не найден.")
        return

    answers = []
    
    for question in tqdm(questions_df['Вопрос'], desc="Генерация ответов"):
        try:
            answer = pipeline.run(question=question)
            answers.append(answer)
        except Exception as e:
            print(f"\n!!! Ошибка при обработке вопроса '{question[:50]}...': {e}")
            # Добавляем заглушку, чтобы не нарушать структуру submission
            answers.append("Произошла ошибка при обработке этого вопроса.")

    # --- 6. Сохранение Результатов ---
    questions_df['Ответы на вопрос'] = answers
    questions_df.to_csv(config.SUBMISSION_PATH, index=False)
    resource_manager.log_checkpoint(f"Файл {config.SUBMISSION_PATH} сгенерирован")


if __name__ == "__main__":
    # Загружаем API-ключи из .env файла
    load_dotenv()
    
    # Запускаем основной процесс
    main_workflow()
    
    # --- 7. Финальный Отчет по Ресурсам ---
    print("\n" + "="*25 + " ФИНАЛЬНЫЙ ОТЧЕТ ПО РЕСУРСАМ " + "="*25)
    summary = resource_manager.get_summary()
    for key, value in summary.items():
        # Форматируем вывод для лучшей читаемости
        if "usd" in key:
            print(f"{key:<20}: ${value:.4f}")
        elif "sec" in key:
            print(f"{key:<20}: {value:.2f} s")
        else:
            print(f"{key:<20}: {value:.2f}")
    print("="*80)