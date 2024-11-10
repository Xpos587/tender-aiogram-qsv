from typing import Final

from aiogram import Router

from . import start, help, validation

router: Final[Router] = Router(name=__name__)
router.include_routers(start.router, help.router, validation.router)
