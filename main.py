"""
Главный исполняемый файл проекта "Надежный Агент-Аналитик".

Этот скрипт является точкой входа и оркестратором всего процесса. Он выполняет
следующие шаги:
1. Настраивает пути для корректной работы импортов.
2. Загружает переменные окружения (API-ключи).
3. Проверяет наличие готовых поисковых индексов и запускает их создание
   при необходимости ("ленивая" инициализация).
4. Собирает полный RAG-пайплайн со всеми компонентами.
5. Запускает обработку вопросов из `questions.csv` в отказоустойчивом режиме.
6. Сохраняет результаты в `submission.csv`.
7. Выводит финальный отчет о потраченных ресурсах.
"""

import sys
import os
import pandas as pd
from tqdm import tqdm
import csv
from dotenv import load_dotenv

# --- 1. Настройка Окружения ---

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. Импорты Наших Модулей (после настройки sys.path) ---
from config import config
from utils.resource_manager import resource_manager
from rag_components.llm_client import LLMClient
from rag_components.agents import SeaAgent, RefinementAgent, DecompositionAgent
from rag_components.retrievers import build_ensemble_retriever
from rag_components.pipeline import RAGPipeline
from prepare_logic import prepare_all_indices


def main_workflow():
    """
    Основная функция, реализующая полный рабочий процесс.
    """
    resource_manager.log_checkpoint("Старт основного рабочего процесса")

    # --- 3. "Ленивая" Инициализация Индексов ---
    
    if not os.path.exists(os.path.join(config.STORAGE_PATH, "all_docs.pkl")):
        print("Индексы не найдены или созданы не полностью. Запускаю процесс создания...")
        prepare_all_indices()
    else:
        resource_manager.log_checkpoint("Обнаружены готовые индексы. Пропускаем этап подготовки.")

    # --- 4. Сборка RAG-Пайплайна ---
    resource_manager.log_checkpoint("Сборка RAG-пайплайна...")
    try:
        llm_client = LLMClient()
        sea_agent = SeaAgent(llm_client=llm_client)
        refinement_agent = RefinementAgent(llm_client=llm_client)
        decomposition_agent = DecompositionAgent(llm_client=llm_client)
        ensemble_retriever = build_ensemble_retriever()

        pipeline = RAGPipeline(
            retriever=ensemble_retriever,
            llm_client=llm_client,
            sea_agent=sea_agent,
            refinement_agent=refinement_agent,
            decomposition_agent = decomposition_agent
        )
        resource_manager.log_checkpoint("RAG-пайплайн успешно собран")
    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при сборке пайплайна: {e}")
        return

    # --- 5. Основной Цикл Обработки Вопросов с Построчной Записью ---
    try:
        questions_df = pd.read_csv(config.QUESTIONS_PATH)
    except FileNotFoundError:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Файл {config.QUESTIONS_PATH} не найден.")
        return

    
    if not os.path.exists(config.SUBMISSION_PATH):
        submission_df = questions_df.copy()
        submission_df['Ответы на вопрос'] = pd.NA
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Создан пустой файл {config.SUBMISSION_PATH}")
    else:
        submission_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Найден существующий файл {config.SUBMISSION_PATH}. Попытка возобновления.")
        
        if 'Ответы на вопрос' not in submission_df.columns:
            submission_df['Ответы на вопрос'] = pd.NA

    
    with tqdm(total=len(questions_df), desc="Генерация ответов") as progress_bar:
        # Возвращаемся к iterrows(), так как он самый надежный по доступу к колонкам
        for index, row in questions_df.iterrows():
            question = str(row['Вопрос'])
            
            # Pylance может ругаться на 'index', но мы знаем, что это int.
            # Используем # type: ignore для подавления ложного срабатывания.
            if index < len(submission_df) and pd.notna(submission_df.loc[index, 'Ответы на вопрос']): # type: ignore
                progress_bar.update(1)
                continue

            try:
                answer = pipeline.run(question=question)
                normalized_answer = answer.replace('\n', ' ').replace('\r', '')
                submission_df.loc[index, 'Ответы на вопрос'] = normalized_answer # type: ignore
            except Exception as e:
                error_message = f"ОШИБКА: {e}"
                print(f"\n!!! Ошибка при обработке вопроса '{question[:50]}...': {error_message}")
                normalized_answer = answer.replace('\n', ' ').replace('\r', '')
                submission_df.loc[index, 'Ответы на вопрос'] = normalized_answer # type: ignore
            
            submission_df.to_csv(
                config.SUBMISSION_PATH, 
                index=False,
                quoting=csv.QUOTE_ALL,
                lineterminator='\n' 
            )
            progress_bar.update(1)

    resource_manager.log_checkpoint(f"Файл {config.SUBMISSION_PATH} полностью сгенерирован")


if __name__ == "__main__":
    load_dotenv()
    main_workflow()
    
    # --- 7. Финальный Отчет по Ресурсам ---
    print("\n" + "="*25 + " ФИНАЛЬНЫЙ ОТЧЕТ ПО РЕСУРСАМ " + "="*25)
    summary = resource_manager.get_summary()
    for key, value in summary.items():
        if "usd" in key:
            print(f"{key:<20}: ${value:.8f}")
        elif "sec" in key:
            print(f"{key:<20}: {value:.2f} s")
        else:
            print(f"{key:<20}: {value:.2f} MB")
    print("="*80)