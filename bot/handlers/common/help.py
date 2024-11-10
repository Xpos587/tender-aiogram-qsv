from typing import Any, Final
from aiogram import Router
from aiogram.types import CallbackQuery, InputMediaPhoto
from aiogram_i18n import I18nContext
from aiogram.fsm.context import FSMContext
from assets import image

from bot.filters import CallbackData as cbd
from bot.keyboards import Button, common_keyboard

router: Final[Router] = Router(name=__name__)


@router.callback_query(cbd.help())
async def help_command(
    query: CallbackQuery,
    i18n: I18nContext,
    state: FSMContext,
) -> Any:
    reply_markup = common_keyboard(
        rows=[
            Button(i18n.btn.back(), callback_data=cbd.main),
        ]
    )

    return await query.message.edit_media(
        InputMediaPhoto(media=image.start, caption=i18n.help()),
        reply_markup=reply_markup,
    )
