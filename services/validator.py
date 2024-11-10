import aiohttp
import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import pymupdf4llm
import tempfile
import os


class Customer(BaseModel):
    name: str
    id: int


class State(BaseModel):
    name: str
    id: int


class Region(BaseModel):
    tree_path_id: str = Field(..., alias="treePathId")
    socr: str
    id: int
    oktmo: str
    code: str
    name: str


class File(BaseModel):
    company_id: Optional[int] = Field(None, alias="companyId")
    name: str
    id: int
    content: Optional[bytes] = None


class DeliveryItem(BaseModel):
    sum: float
    cost_per_unit: float = Field(..., alias="costPerUnit")
    quantity: float
    name: str
    buyer_id: Optional[int] = Field(None, alias="buyerId")
    is_buyer_invitation_sent: bool = Field(..., alias="isBuyerInvitationSent")
    is_approved_by_buyer: Optional[bool] = Field(
        None, alias="isApprovedByBuyer"
    )


class Delivery(BaseModel):
    period_days_from: Optional[int] = Field(None, alias="periodDaysFrom")
    period_days_to: Optional[int] = Field(None, alias="periodDaysTo")
    period_date_from: Optional[str] = Field(None, alias="periodDateFrom")
    period_date_to: Optional[str] = Field(None, alias="periodDateTo")
    delivery_place: str = Field(..., alias="deliveryPlace")
    quantity: float
    items: List[DeliveryItem]
    id: int


class AuctionItem(BaseModel):
    current_value: float = Field(..., alias="currentValue")
    cost_per_unit: float = Field(..., alias="costPerUnit")
    okei_name: str = Field(..., alias="okeiName")
    created_offer_id: Optional[int] = Field(None, alias="createdOfferId")
    sku_id: Optional[int] = Field(None, alias="skuId")
    image_id: Optional[int] = Field(None, alias="imageId")
    default_image_id: Optional[int] = Field(None, alias="defaultImageId")
    okpd_name: str = Field(..., alias="okpdName")
    production_directory_name: str = Field(..., alias="productionDirectoryName")
    oksm: Optional[str]
    name: Optional[str]
    id: int


class Bet(BaseModel):
    num: int
    cost: float
    server_time: str = Field(..., alias="serverTime")
    is_auto_bet: bool = Field(..., alias="isAutoBet")
    auction_id: int = Field(..., alias="auctionId")
    supplier_id: int = Field(..., alias="supplierId")
    create_user_id: int = Field(..., alias="createUserId")
    last_manual_server_time: Optional[str] = Field(
        None, alias="lastManualServerTime"
    )
    id: int


class TenderData(BaseModel):
    customer: Customer
    created_by_customer: Customer = Field(..., alias="createdByCustomer")
    state: State
    start_date: str = Field(..., alias="startDate")
    initial_duration: float = Field(..., alias="initialDuration")
    end_date: str = Field(..., alias="endDate")
    start_cost: float = Field(..., alias="startCost")
    next_cost: Optional[float] = Field(None, alias="nextCost")
    last_bet_cost: Optional[float] = Field(None, alias="lastBetCost")
    step: float
    auction_item: List[AuctionItem] = Field(..., alias="auctionItem")
    bets: List[Bet]
    unique_supplier_count: int = Field(..., alias="uniqueSupplierCount")
    auction_region: List[Region] = Field(..., alias="auctionRegion")
    repeat_id: Optional[int] = Field(None, alias="repeatId")
    unpublish_name: Optional[str] = Field(None, alias="unpublishName")
    unpublish_date: Optional[str] = Field(None, alias="unpublishDate")
    federal_law_name: str = Field(..., alias="federalLawName")
    conclusion_reason_name: Optional[str] = Field(
        None, alias="conclusionReasonName"
    )
    items: List[AuctionItem]
    deliveries: List[Delivery]
    files: List[File]
    license_files: List[File] = Field(..., alias="licenseFiles")
    is_electronic_contract_execution_required: bool = Field(
        ..., alias="isElectronicContractExecutionRequired"
    )
    is_contract_guarantee_required: bool = Field(
        ..., alias="isContractGuaranteeRequired"
    )
    contract_guarantee_amount: Optional[float] = Field(
        None, alias="contractGuaranteeAmount"
    )
    is_license_production: bool = Field(..., alias="isLicenseProduction")
    name: str
    id: int


class ProxyManager:
    def __init__(self):
        # Формируем строку для прокси
        self.proxy = "http://7oqrVE:Sk6VrR@193.187.147.133:8000"
        self.use_proxy = False

    def toggle_proxy(self):
        """Включение/выключение прокси"""
        self.use_proxy = not self.use_proxy
        return self.use_proxy

    def get_proxy(self):
        """Получение текущих настроек прокси"""
        return self.proxy if self.use_proxy else None


class TenderParser:
    def __init__(self):
        self.base_url = "https://zakupki.mos.ru/newapi/api/Auction/Get"
        self.file_url = "https://zakupki.mos.ru/newapi/api/FileStorage/Download"
        self.headers = {"accept": "application/json"}
        self.proxy_manager = ProxyManager()

    async def convert_doc_to_markdown(
        self, content: bytes, filename: str
    ) -> str:
        try:
            ext = filename.lower().split(".")[-1]

            # Создаем временную директорию для файлов
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input_path = os.path.join(temp_dir, f"input.{ext}")

                # Записываем входной файл
                with open(temp_input_path, "wb") as f:
                    f.write(content)

                if ext == "pdf":
                    # Прямое преобразование PDF в Markdown
                    return pymupdf4llm.to_markdown(temp_input_path)

                elif ext in ("doc", "docx"):
                    # Конвертация в PDF с помощью LibreOffice
                    process = await asyncio.create_subprocess_exec(
                        "libreoffice",
                        "--headless",
                        "--convert-to",
                        "pdf",
                        temp_input_path,
                        "--outdir",
                        temp_dir,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    stdout, stderr = await process.communicate()

                    if process.returncode != 0:
                        raise Exception(
                            f"Conversion failed: {
                                stderr.decode()}"
                        )

                    # Определяем имя выходного PDF-файла (оно будет иметь то же имя, что и оригинал)
                    temp_output_pdf = temp_input_path.replace(f".{ext}", ".pdf")

                    if not os.path.exists(temp_output_pdf):
                        raise FileNotFoundError(
                            f"No such file: '{temp_output_pdf}'"
                        )

                    # Преобразование PDF в Markdown с использованием pymupdf4llm
                    return pymupdf4llm.to_markdown(
                        temp_output_pdf,
                        write_images=False,
                        embed_images=False,
                        graphics_limit=None,
                        margins=(0, 0, 0, 0),
                        table_strategy="lines_strict",
                        fontsize_limit=1,
                        ignore_code=True,
                        show_progress=False,
                    )

        except Exception as e:
            print(f"Error converting {filename}: {str(e)}")
            return None

    async def fetch_tender(
        self, session: aiohttp.ClientSession, auction_id: int
    ) -> Dict:
        params = {"auctionId": auction_id}
        proxy = self.proxy_manager.get_proxy()
        async with session.get(
            self.base_url, params=params, headers=self.headers, proxy=proxy
        ) as response:
            if response.status == 200:
                data = await response.json()
                return TenderData.model_validate(data)
            return None

    async def get_file_bytes(self, file_id: int) -> bytes:
        proxy = self.proxy_manager.get_proxy()
        async with aiohttp.ClientSession() as session:
            params = {"id": file_id}
            async with session.get(
                self.file_url, params=params, proxy=proxy
            ) as response:
                if response.status == 200:
                    return await response.read()
                return None

    async def process_tenders(
        self, auction_ids: List[int], get_files: bool = False
    ) -> List[TenderData]:
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_tender(session, aid) for aid in auction_ids]
            tenders = [r for r in await asyncio.gather(*tasks) if r is not None]

            if get_files:
                for tender in tenders:
                    for file in tender.files:
                        file.content = await self.get_file_bytes(file.id)
                        if file.content and file.name.lower().endswith(
                            (".doc", ".docx", ".pdf")
                        ):
                            markdown_content = (
                                await self.convert_doc_to_markdown(
                                    file.content, file.name
                                )
                            )
                            if markdown_content:
                                file.content = markdown_content

            return tenders


class TenderValidator:
    tender: TenderData

    def __init__(self, tender: TenderData):
        self.tender = tender
        self.technical_task = self._get_technical_task()
        self.contract = self._get_contract()

    def _get_technical_task(self):
        """Заглушка для получения ТЗ"""
        # Здесь можно будет реализовать логику извлечения ТЗ из файлов
        return None

    def _get_contract(self):
        """Заглушка для получения проекта контракта"""
        # Здесь можно будет реализовать логику извлечения контракта из файлов
        return None

    def validate_name(self) -> bool:
        """
        1. Проверка соответствия наименования закупки.
        Заглушка: возвращает всегда False.
        """
        return False

    def validate_contract_guarantee(self) -> bool:
        """
        2. Проверка требования обеспечения контракта.
        Заглушка: возвращает всегда False.
        """
        return False

    def validate_certificates(self) -> bool:
        """
        3. Проверка требований к сертификатам/лицензиям.
        Заглушка: возвращает всегда False.
        """
        return False

    def validate_delivery_schedule(self) -> bool:
        """
        4. Проверка графика и этапов поставки.
        Заглушка: возвращает всегда False.
        """
        return False

    def validate_price(self) -> bool:
        """
        5. Проверка цены контракта.
        Заглушка: возвращает всегда False.
        """
        return False

    def validate_specifications(self) -> bool:
        """
        6. Проверка спецификаций в техническом задании.
        Заглушка: возвращает всегда False.
        """
        return False
