"""
Скрипт для сборки проекта в один монолитный файл `main.py` для сдачи.

Этот скрипт выполняет следующие действия:
1. Определяет правильный порядок файлов для конкатенации, чтобы избежать
   ошибок с зависимостями (классы и функции определяются до их использования).
2. Читает содержимое каждого файла.
3. Отделяет импорты от основного кода.
4. Собирает все внешние импорты в единый блок в начале файла, удаляя дубликаты.
5. Игнорирует все локальные импорты (например, `from rag_components...`), так как
   в монолитном файле они больше не нужны.
6. Последовательно записывает код из каждого файла в один выходной файл,
   разделяя их комментариями для читаемости.
"""

import os

# 1. ОПРЕДЕЛЯЕМ ПРАВИЛЬНЫЙ ПОРЯДОК ФАЙЛОВ
# Порядок критически важен: от базовых утилит к высокоуровневой логике.
FILE_ORDER = [
    # Базовые утилиты и конфигурация
    'config.py',
    'utils/resource_manager.py',
    'utils/prompt_library.py',
    
    # Низкоуровневые компоненты RAG
    'rag_components/llm_client.py',
    'rag_components/embeddings.py', # Зависит от llm_client
    
    # Компоненты среднего уровня
    'rag_components/retrievers.py', # Зависит от embeddings
    'rag_components/agents.py',     # Зависит от llm_client и prompt_library
    
    # Высокоуровневая логика
    'rag_components/pipeline.py',   # Зависит от agents и retrievers
    'prepare_logic.py',             # Зависит почти от всего
    
    # Главный исполняемый блок
    'main.py'
]

# Префиксы локальных импортов, которые нужно удалить
LOCAL_IMPORT_PREFIXES = (
    'from config',
    'from utils',
    'from rag_components',
    'from prepare_logic'
)

# Имя выходного файла. Мы создаем новый, чтобы не затереть наш рабочий main.py
OUTPUT_FILENAME = 'submission_main.py'


def build():
    """
    Основная функция сборки.
    """
    print("🚀 Начало сборки проекта в монолитный файл...")
    
    all_imports = set()
    file_contents = []

    # --- Шаг 1: Читаем все файлы и разделяем импорты и код ---
    for filename in FILE_ORDER:
        if not os.path.exists(filename):
            print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Файл {filename} не найден, пропускаю.")
            continue

        print(f"   - Обработка файла: {filename}")
        with open(filename, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            
            imports = {
                line for line in lines 
                if line.strip().startswith('import ') or line.strip().startswith('from ')
            }
            
            code = [line for line in lines if line not in imports]
            
            # Фильтруем локальные импорты
            external_imports = {
                imp for imp in imports 
                if not imp.strip().startswith(LOCAL_IMPORT_PREFIXES)
            }
            
            all_imports.update(external_imports)
            file_contents.append((filename, code))

    # --- Шаг 2: Записываем всё в один файл ---
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
        # Записываем заголовок
        outfile.write("# ======================================================================\n")
        outfile.write("#  ЭТОТ ФАЙЛ СГЕНЕРИРОВАН АВТОМАТИЧЕСКИ. НЕ РЕДАКТИРУЙТЕ ЕГО ВРУЧНУЮ.\n")
        outfile.write("# ======================================================================\n\n")

        # Записываем все уникальные внешние импорты в начало
        outfile.write("# === ГЛОБАЛЬНЫЕ ИМПОРТЫ ===\n")
        outfile.writelines(sorted(list(all_imports)))
        outfile.write("\n\n")

        # Последовательно записываем код из каждого файла
        for filename, code in file_contents:
            outfile.write(f"# {'='*20} ИЗ ФАЙЛА: {filename} {'='*20}\n\n")
            outfile.writelines(code)
            outfile.write("\n\n")

    print(f"✅ Проект успешно собран в файл: {OUTPUT_FILENAME}")
    print("Теперь вы можете переименовать его в 'main.py' и добавить в zip-архив для сдачи.")


if __name__ == "__main__":
    build()