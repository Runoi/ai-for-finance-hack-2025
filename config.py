"""
Центральный конфигурационный файл для всего RAG-проекта.

Этот файл содержит все основные настройки, пути к файлам и "тумблеры" 
для управления поведением архитектуры. Изменение параметров здесь 
позволяет гибко настраивать пайплайн без изменения основного кода.
"""

class Config:
    """Класс-контейнер для всех настроек проекта."""
    
    # === ПУТИ К ФАЙЛАМ И ПАПКАМ ===
    TRAIN_DATA_PATH = "./train_data.csv"
    QUESTIONS_PATH = "./questions.csv"
    SUBMISSION_PATH = "submission.csv"
    STORAGE_PATH = "storage"
    
    # === НАСТРОЙКИ МОДЕЛЕЙ ===
    # Модели для разных ролей в агентской системе
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    GENERATOR_MODEL = "openrouter/google/gemma-3-27b-it"
    ANALYST_MODEL   = "openrouter/meta-llama/llama-3-70b-instruct"
    REFINER_MODEL   = "openrouter/meta-llama/llama-3-70b-instruct"
    # Для Mistral: "openrouter/mistralai/mistral-small-3.2-24b-instruct"
    # Для Grok: "openrouter/x-ai/grok-3-mini"

    # === НАСТРОЙКИ РЕТРИВЕРА ===
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    # Веса для EnsembleRetriever [Семантика, Ключи, Мета, Граф]
    RETRIEVER_WEIGHTS = [0.4, 0.25, 0.2, 0.15]

    # === "ТУМБЛЕРЫ" ДЛЯ АРХИТЕКТУРЫ ===
    # Включение/отключение "голов" ретривера для экспериментов
    ENABLE_SEMANTIC_HEAD = True
    ENABLE_KEYWORD_HEAD = True
    ENABLE_METADATA_HEAD = True
    ENABLE_GRAPH_CONCEPT_HEAD = True

    MODEL_PRICES = {
        # Генеративные модели (цена за Input / Output)
        "openrouter/google/gemma-3-27b-it": {"input": 0.09, "output": 0.16},
        "openrouter/meta-llama/llama-3-70b-instruct": {"input": 0.30, "output": 0.40},
        "openrouter/mistralai/mistral-small-3.2-24b-instruct": {"input": 0.06, "output": 0.18},
        "openrouter/x-ai/grok-3-mini": {"input": 0.30, "output": 0.50},

        # Эмбеддинг модели (цена только за Input)
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    }

    # === НАСТРОЙКИ АГЕНТСКОГО ЦИКЛА ===
    # Максимальное количество итераций для FAIR-RAG цикла.
    # 1 = простой RAG. 2-3 = оптимально для сложных вопросов.
    MAX_ITERATIONS = 2

    MAX_CONTEXT_DOCS = 15

# Глобальный экземпляр конфига для импорта в других модулях
config = Config()