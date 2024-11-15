start =
    🚤 Катирок Бот приветствует вас!

    🤖 Я помогу проверить корректность котировочных сессий и выявить возможные нарушения в их оформлении.

    👇 Выберите нужное действие:

help =
    ℹ️ Справка по работе с ботом

    🔷 Основные функции:

    1. 🔍 Проверка котировочных сессий:
       • Отправка одной или нескольких ссылок на КС
       • Автоматическая проверка по настроенным правилам
       • Получение подробного отчета о результатах

    2. ⚙️ Настройка правил проверки:
       • Включение/отключение отдельных правил
       • Гибкая настройка под ваши требования
       • Сохранение настроек между сессиями

    🔷 Правила проверки:

    1. Проверка наименования закупки
    2. Проверка обеспечения контракта
    3. Проверка сертификатов/лицензий
    4. Проверка графика поставки
    5. Проверка цены контракта
    6. Проверка спецификаций

    🔷 Как использовать:

    1. Нажмите "🔍 Проверить ссылку"
    2. Отправьте одну или несколько ссылок на КС
    3. Дождитесь результатов проверки
    4. Используйте "⚙️ Настройки проверок" для управления правилами

    💡 Подсказка: Можно отправлять несколько ссылок, каждую с новой строки

enter_link =
    📎 Отправьте ссылку на тендер для проверки

    Можно отправить несколько ссылок, каждая с новой строки

    Формат: https://zakupki.mos.ru/auction/НОМЕР

invalid_url_format =
    ❌ Неверный формат ссылки

    Пример правильной ссылки:
    https://zakupki.mos.ru/auction/9864533

tender_not_found =
    ❌ Тендер не найден

    Возможные причины:
    • Тендер удален
    • Нет доступа к тендеру
    • Технические проблемы

validation_error =
    ❌ Ошибка при проверке тендера

    Пожалуйста, попробуйте позже или обратитесь к администратору.

validation_settings = 🔧 Настройки валидации
validation_name = Проверка наименования
validation_guarantee = Проверка обеспечения контракта
validation_certificates = Проверка сертификатов/лицензий
validation_delivery = Проверка графика поставки
validation_price = Проверка цены контракта
validation_specs = Проверка спецификаций

validation_status_enabled = ✅ Включено
validation_status_disabled = ❌ Выключено

processing_in_progress =
    ⏳ Идет обработка предыдущего запроса...

    Пожалуйста, дождитесь завершения текущей проверки.

tender_validation = 📋 Результаты проверки тендера №<a href='https://zakupki.mos.ru/auction/{$tender_id}'>№{$tender_id}</a>

    {$unpublish_reason}

    {$results}

unpublish_reason = ❗️ Причина снятия: {$reason}

