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
    
      # === ГЛАВНЫЙ СТРАТЕГИЧЕСКИЙ ПЕРЕКЛЮЧАТЕЛЬ ===
    # - 'ITERATIVE': Поиск -> Анализ (SEA) -> Уточнение (Refinement).
    # - 'DECOMPOSE': Декомпозиция -> Поиск -> Генерация.
    # - 'SIMPLE':    Поиск -> Генерация (самый быстрый режим).
    STRATEGY: str = 'ITERATIVE' 

    # === НАСТРОЙКИ МОДЕЛЕЙ ===
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    GENERATOR_MODEL: str = "openrouter/mistralai/mistral-small-3.2-24b-instruct"
    ANALYST_MODEL: str = "openrouter/meta-llama/llama-3-70b-instruct"
    REFINER_MODEL: str = "openrouter/meta-llama/llama-3-70b-instruct"
    DECOMPOSER_MODEL: str = "openrouter/meta-llama/llama-3-70b-instruct"
    #"openrouter/meta-llama/llama-3-70b-instruct"
    #"openrouter/google/gemma-3-27b-it"
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

    # --- ПАРАМЕТРЫ ДЛЯ MMR ---
    # Количество документов, которые нужно сначала найти (кандидаты)
    MMR_FETCH_K = 50 
    # Количество документов, которые нужно вернуть после MMR-обработки
    MMR_K = 20
    # Коэффициент разнообразия (0.0 - макс. релевантность, 1.0 - макс. разнообразие)
    MMR_LAMBDA_MULT = 0.5 

    # === НАСТРОЙКИ АГЕНТСКОГО ЦИКЛА ===
    # Максимальное количество итераций для FAIR-RAG цикла.
    # 1 = простой RAG. 2-3 = оптимально для сложных вопросов.
    MAX_ITERATIONS = 1

    MAX_CONTEXT_DOCS = 10

    ENABLE_PARALLEL_REQUESTS: bool = True

# Глобальный экземпляр конфига для импорта в других модулях
config = Config()