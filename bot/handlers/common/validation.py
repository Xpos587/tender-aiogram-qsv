import re
import logging
from typing import Any, Final, List
from aiogram import Router
from aiogram.types import CallbackQuery, Message, InputMediaPhoto
from aiogram.fsm.context import FSMContext
from aiogram_i18n import I18nContext
from assets import image
from services.validator import TenderParser, TenderValidator

from bot.filters import CallbackData as cbd, ChatStates
from bot.keyboards import Button, common_keyboard

router: Final[Router] = Router(name=__name__)
logger = logging.getLogger(__name__)

# Начальные условия валидации
VALIDATION_CONDITIONS = {
    "name": ["validation_name", True],
    "guarantee": ["validation_guarantee", True],
    "certificates": ["validation_certificates", True],
    "delivery": ["validation_delivery", True],
    "price": ["validation_price", True],
    "specs": ["validation_specs", True],
}


def extract_tender_ids(text: str) -> List[int]:
    """Извлекает ID тендеров из текста со ссылками"""
    pattern = r"https://zakupki\.mos\.ru/auction/(\d+)"
    return [int(match.group(1)) for match in re.finditer(pattern, text)]


async def validate_tenders(
    tender_ids: List[int], conditions: dict
) -> List[dict]:
    """Выполняет валидацию нескольких тендеров согласно настроенным условиям"""
    parser = TenderParser()
    tenders = await parser.process_tenders(tender_ids, get_files=True)

    if not tenders:
        return []

    validation_methods = {
        "name": lambda v: v.validate_name(),
        "guarantee": lambda v: v.validate_contract_guarantee(),
        "certificates": lambda v: v.validate_certificates(),
        "delivery": lambda v: v.validate_delivery_schedule(),
        "price": lambda v: v.validate_price(),
        "specs": lambda v: v.validate_specifications(),
    }

    results = []
    for tender in tenders:
        validator = TenderValidator(tender)
        tender_results = {}

        for key, (text_key, enabled) in conditions.items():
            if enabled and key in validation_methods:
                tender_results[text_key] = validation_methods[key](validator)

        results.append({"tender_id": tender.id, "results": tender_results})

    return results


@router.callback_query(cbd.validation_settings())
async def show_validation_settings(
    query: CallbackQuery,
    i18n: I18nContext,
    state: FSMContext,
) -> Any:
    data = await state.get_data()
    conditions = data.get("validation_conditions", VALIDATION_CONDITIONS.copy())

    if not data.get("validation_conditions"):
        await state.update_data(validation_conditions=conditions)

    buttons = []
    for key, (text_key, enabled) in conditions.items():
        status = (
            "validation_status_enabled"
            if enabled
            else "validation_status_disabled"
        )
        buttons.append(
            Button(
                f"{i18n.get(text_key)} ({i18n.get(status)})",
                callback_data=cbd.toggle_rule.extend(key),
            )
        )

    buttons.append(Button(i18n.btn.back(), callback_data=cbd.main))
    reply_markup = common_keyboard(rows=buttons)

    return await query.message.edit_media(
        media=InputMediaPhoto(
            media=image.settings, caption=i18n.validation_settings()
        ),
        reply_markup=reply_markup,
    )


@router.callback_query(cbd.toggle_rule())
async def toggle_rule(
    query: CallbackQuery,
    callback_data: str,
    state: FSMContext,
    i18n: I18nContext,
) -> Any:
    data = await state.get_data()
    conditions = data.get("validation_conditions", VALIDATION_CONDITIONS.copy())

    key = callback_data
    if key in conditions:
        text_key, enabled = conditions[key]
        conditions[key] = [text_key, not enabled]
        await state.update_data(validation_conditions=conditions)

    return await show_validation_settings(query, i18n, state)


@router.callback_query(cbd.check_link())
async def process_link_start(
    query: CallbackQuery,
    i18n: I18nContext,
    state: FSMContext,
) -> Any:
    data = await state.get_data()
    validation_conditions = data.get("validation_conditions")

    await state.clear()

    if validation_conditions:
        await state.update_data(validation_conditions=validation_conditions)

    await state.set_state(ChatStates.ReadyToRespond)

    return await query.message.edit_media(
        media=InputMediaPhoto(media=image.checker, caption=i18n.enter_link()),
        reply_markup=common_keyboard(
            [Button(i18n.btn.cancel(), callback_data=cbd.main)]
        ),
    )


@router.message(ChatStates.ReadyToRespond)
async def process_links(
    message: Message,
    i18n: I18nContext,
    state: FSMContext,
) -> Any:
    data = await state.get_data()

    if data.get("is_processing"):
        return await message.answer(i18n.processing_in_progress())

    await state.update_data(is_processing=True)
    status_message = await message.answer("⌛")

    try:
        tender_ids = extract_tender_ids(message.text.strip())

        if not tender_ids:
            await status_message.delete()
            return await message.answer(
                i18n.invalid_url_format(),
                reply_markup=common_keyboard(
                    [Button(i18n.btn.back(), callback_data=cbd.main)]
                ),
            )

        conditions = data.get(
            "validation_conditions", VALIDATION_CONDITIONS.copy()
        )
        validation_results = await validate_tenders(tender_ids, conditions)

        if not validation_results:
            await message.answer(
                i18n.tender_not_found(),
                reply_markup=common_keyboard(
                    [Button(i18n.btn.back(), callback_data=cbd.main)]
                ),
            )
            return

        for result in validation_results:
            tender_id = str(result["tender_id"]).replace(" ", "")
            validation_text = "\n".join(
                f"{i18n.get(key)}: {'✅' if value else '❌'}"
                for key, value in result["results"].items()
            )

            await message.answer(
                i18n.tender_validation(
                    tender_id=tender_id, results=validation_text
                ),
                reply_markup=common_keyboard(
                    [Button(i18n.btn.back(), callback_data=cbd.main)]
                ),
            )

    except Exception as e:
        logger.exception(
            "Error processing tender links for user %s: %s",
            message.from_user.id,
            str(e),
        )
        await message.answer(
            i18n.validation_error(),
            reply_markup=common_keyboard(
                [Button(i18n.btn.back(), callback_data=cbd.main)]
            ),
        )

    finally:
        await state.update_data(is_processing=False)
