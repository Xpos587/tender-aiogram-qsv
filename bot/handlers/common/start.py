from typing import Any, Final
from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, InputMediaPhoto, CallbackQuery
from aiogram_i18n import I18nContext
from assets import image

from bot.filters import CallbackData as cbd, ChatStates
from bot.keyboards import Button, common_keyboard

router: Final[Router] = Router(name=__name__)


@router.message(Command("start"))
@router.callback_query(cbd.main())
async def start_command(
    object: CallbackQuery | Message,
    i18n: I18nContext,
    state: FSMContext,
) -> Any:
    # Получаем текущие настройки валидации из состояния
    data = await state.get_data()
    validation_conditions = data.get("validation_conditions")

    # Очищаем состояние и восстанавливаем настройки валидации
    await state.clear()
    if validation_conditions:
        await state.update_data(validation_conditions=validation_conditions)

    reply_markup = common_keyboard(
        rows=[
            (
                Button(i18n.btn.check_link(), callback_data=cbd.check_link),
                Button(
                    i18n.btn.validation_settings(),
                    callback_data=cbd.validation_settings,
                ),
            ),
            (
                Button(i18n.btn.help(), callback_data=cbd.help),
                Button(
                    i18n.btn.source(),
                    url="https://github.com/xpos587/tender-aiogram-qsv",
                ),
            ),
        ]
    )

    if isinstance(object, CallbackQuery):
        message: Message = getattr(object, "message")
        return await message.edit_media(
            media=InputMediaPhoto(media=image.start, caption=i18n.start()),
            reply_markup=reply_markup,
        )

    message = object
    return await message.answer_photo(
        photo=image.start, caption=i18n.start(), reply_markup=reply_markup
    )
