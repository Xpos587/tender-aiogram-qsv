# tender-aiogram-qsv

Telegram-бот для автоматизированной проверки котировочных сессий на портале [zakupki.mos.ru](https://zakupki.mos.ru/).

## Возможности

- Анализ одной или нескольких котировочных сессий по ссылкам
- Проверка соответствия по 6 основным критериям:
  1. Соответствие наименования закупки в документах
  2. Наличие требований об обеспечении контракта
  3. Проверка требований к сертификатам/лицензиям
  4. Соответствие графика поставки
  5. Указание цены контракта
  6. Наличие спецификаций в техническом задании
- Настройка правил проверки (включение/отключение отдельных критериев)
- Поддержка обработки нескольких ссылок одновременно
- Автоматическое определение типов документов (ТЗ, контракт)

## Технологии

- Python 3.12
- aiogram 3.x
- PyTorch
- Transformers
- sentence-transformers
- PostgreSQL
- Redis
- Docker

## ML модели

- Question-Answering: `timpal0l/mdeberta-v3-base-squad2`
- Embeddings: `sergeyzh/rubert-tiny-turbo`
- Zero-shot классификация: `cointegrated/rubert-base-cased-nli-threeway`

## Установка и запуск

```bash
# Клонирование репозитория
git clone https://github.com/xpos587/tender-aiogram-qsv.git
cd tender-aiogram-qsv

# Запуск через Docker Compose
docker compose up --build -d
```

## Архитектура

```
.
├── alembic/          # Миграции базы данных
├── assets/           # Статические файлы
├── bot/
│   ├── filters/      # Фильтры сообщений
│   ├── handlers/     # Обработчики команд
│   ├── keyboards/    # Клавиатуры
│   └── middlewares/  # Промежуточные обработчики
├── lang/             # Локализация (FTL)
├── services/
│   ├── database/     # Работа с БД
│   └── validator.py  # Валидация тендеров
└── utils/            # Вспомогательные функции
```

## Особенности реализации

- Асинхронная обработка запросов через aiohttp
- Конвертация документов (DOC, DOCX, XLSX → PDF → Markdown)
- Использование GPU для ML моделей (если доступен)
- Поддержка прокси для обхода ограничений API
- Сохранение пользовательских настроек в PostgreSQL
- Локализация через Fluent (FTL)

## Лицензия

MIT

## Авторы

- [Michael](https://github.com/xpos587)

