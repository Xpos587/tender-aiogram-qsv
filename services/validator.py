from scipy.spatial.distance import cosine
import Levenshtein
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import numpy as np
from typing import List, Tuple
import logging
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
    contract_cost: Optional[float] = Field(None, alias="contractCost")
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
    upload_license_documents_comment: Optional[str] = Field(
        None, alias="uploadLicenseDocumentsComment"
    )
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

                elif ext in ("doc", "docx", "xlsx"):
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
                            (".doc", ".docx", ".xlsx", ".pdf")
                        ):
                            markdown_content = (
                                await self.convert_doc_to_markdown(
                                    file.content, file.name
                                )
                            )
                            if markdown_content:
                                file.content = markdown_content

            return tenders


logger = logging.getLogger(__name__)

# Константы для моделей
QA_MODEL_NAME = "timpal0l/mdeberta-v3-base-squad2"
EMBEDDINGS_MODEL_NAME = "sergeyzh/rubert-tiny-turbo"
ZERO_SHOT_MODEL_NAME = "cointegrated/rubert-base-cased-nli-threeway"

# Определение устройства
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Инициализация моделей
qa_model = pipeline(
    "question-answering",
    QA_MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1,
)
embeddings_model = SentenceTransformer(EMBEDDINGS_MODEL_NAME).to(DEVICE)
zero_shot_tokenizer = AutoTokenizer.from_pretrained(ZERO_SHOT_MODEL_NAME)
zero_shot_model = AutoModelForSequenceClassification.from_pretrained(
    ZERO_SHOT_MODEL_NAME
).to(DEVICE)

# Константы для проверок
SIMILARITY_THRESHOLD_HIGH = 0.93
SIMILARITY_THRESHOLD_LOW = 0.8
CER_THRESHOLD_HIGH = 0.97
CER_THRESHOLD_LOW = 0.8
ZERO_SHOT_THRESHOLD = 0.95
DOCUMENT_NAMES = ["техническом задании", "контракте"]


def calculate_character_error_rate(reference: str, hypothesis: str) -> float:
    """Вычисляет нормализованное расстояние Левенштейна между строками"""
    char_lev_dist = Levenshtein.distance(reference, hypothesis)
    return char_lev_dist / max(len(reference), len(hypothesis))


def predict_zero_shot(
    text: str,
    label_texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    label: str = "entailment",
    normalize: bool = True,
) -> np.ndarray:
    """Выполняет zero-shot классификацию текста"""
    tokens = tokenizer(
        [text] * len(label_texts),
        label_texts,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    # Перемещаем токены на нужное устройство
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.inference_mode():
        result = torch.softmax(model(**tokens).logits, -1)
    proba = result[:, model.config.label2id[label]].cpu().numpy()
    if normalize:
        proba /= sum(proba)
    return proba


class TenderValidator:
    def _classify_documents(self) -> Tuple[List[bool], List[bool]]:
        """Классифицирует документы, определяя технические задания и контракты"""
        tech_specs = []
        contracts = []

        for file in self.tender.files:
            text = file.content.lower()
            text = text.replace("__", "").replace("  ", "")
            first_sentences = " ".join(text.split("\n")[:10])
            filename = file.name.lower()

            # Определение ТЗ по имени или содержимому
            is_tech_spec = (
                ("техническ" in filename and "задан" in filename)
                or ("тз" in filename)
                or (
                    "техническ" in first_sentences
                    and "задан" in first_sentences
                )
            )
            tech_specs.append(is_tech_spec)

            # Определение контракта по имени или содержимому
            is_contract = ("контракт" in filename or "договор" in filename) or (
                ("контракт" in first_sentences or "договор" in first_sentences)
                and (
                    "предмет" in first_sentences or "стороны" in first_sentences
                )
            )
            contracts.append(is_contract)

            logger.debug(
                f"Document '{file.name}': "
                f"{'is' if is_tech_spec else 'is not'} technical specification, "
                f"{'is' if is_contract else 'is not'} contract"
            )

        return tech_specs, contracts

    def _get_technical_task(self) -> List[str]:
        """Извлекает содержимое технического задания"""
        tech_spec_idx = (
            self.tech_spec_indices.index(True)
            if True in self.tech_spec_indices
            else 1
        )
        text = self.tender.files[tech_spec_idx].content
        text = text.replace("__", "").replace("  ", "")
        return [
            sentence.strip()
            for sentence in text.split("\n")
            if sentence.strip()
        ]

    def _get_contract(self) -> List[str]:
        """Извлекает содержимое контракта"""
        contract_idx = (
            self.contract_indices.index(True)
            if True in self.contract_indices
            else 0
        )
        text = self.tender.files[contract_idx].content
        text = text.replace("__", "").replace("  ", "")
        return [
            sentence.strip()
            for sentence in text.split("\n")
            if sentence.strip()
        ]

    def __init__(self, tender: TenderData):
        self.qa_model = qa_model
        self.embeddings_model = embeddings_model
        self.zero_shot_tokenizer = zero_shot_tokenizer
        self.zero_shot_model = zero_shot_model
        self.tender = tender

        # Классификация документов
        self.tech_spec_indices, self.contract_indices = (
            self._classify_documents()
        )

        if not any(self.tech_spec_indices):
            logger.warning("1️⃣ Техническое задание не найдено в документах")
        if not any(self.contract_indices):
            logger.warning("2️⃣ Контракт не найден в документах")

        # Получение документов
        self.technical_task = self._get_technical_task()
        self.contract = self._get_contract()

    def validate_name(self) -> bool:
        """Проверяет соответствие наименования в документах"""
        name_matches = False

        for doc_idx, doc_text in enumerate(
            [self.technical_task, self.contract]
        ):
            doc_content = ". ".join(doc_text[:7])
            extracted_name = self.qa_model(
                question="Какое Наименование закупки?", context=doc_content
            )["answer"]

            cer_score = calculate_character_error_rate(
                self.tender.name, extracted_name
            )
            embeddings = self.embeddings_model.encode(
                [self.tender.name, extracted_name]
            )
            similarity_score = (
                1 - cosine(embeddings[0], embeddings[1]) + 1
            ) / 2

            if (
                cer_score > CER_THRESHOLD_HIGH
                and similarity_score > SIMILARITY_THRESHOLD_HIGH
            ):
                name_matches = True
            elif (
                similarity_score > SIMILARITY_THRESHOLD_LOW
                and cer_score > CER_THRESHOLD_LOW
            ):
                name_matches = True
                logger.warning(
                    f"Частичное несоответствие наименования в {
                        DOCUMENT_NAMES[doc_idx]}"
                )

        return name_matches

    def validate_contract_guarantee(self) -> bool:
        """
        Проверяет наличие требований об обеспечении контракта в документах.
        Выполняется только если обеспечение контракта требуется.
        """
        if not self.tender.is_contract_guarantee_required:
            return True

        GUARANTEE_KEYWORDS = ["требуе", "исполн", "контракт"]
        GUARANTEE_CLASSES = [
            "Требуется обеспечение исполнения контракта",
            "Другое",
        ]
        PROBABILITY_THRESHOLD = 0.95

        guarantee_mentioned = False

        for doc_idx, doc_text in enumerate(
            [self.technical_task, self.contract]
        ):
            matching_sentences = []
            for sentence in doc_text:
                sentence_lower = sentence.lower()
                has_requirement = GUARANTEE_KEYWORDS[0] in sentence_lower
                has_execution = GUARANTEE_KEYWORDS[1] in sentence_lower
                has_contract = GUARANTEE_KEYWORDS[2] in sentence_lower

                if (
                    (has_requirement and has_contract)
                    or (has_execution and has_contract)
                    or (has_requirement and has_execution and has_contract)
                ):
                    matching_sentences.append(sentence.strip())

            for sentence in matching_sentences:
                probability = predict_zero_shot(
                    sentence,
                    GUARANTEE_CLASSES,
                    self.zero_shot_model,
                    self.zero_shot_tokenizer,
                )[0]

                if probability > PROBABILITY_THRESHOLD:
                    guarantee_mentioned = True
                    break

            if not guarantee_mentioned:
                logger.warning(
                    f"Требование об обеспечении контракта не найдено в {
                        DOCUMENT_NAMES[doc_idx]}"
                )

        return guarantee_mentioned

    def validate_certificates(self) -> bool:
        """
        Проверяет наличие требований к лицензиям/сертификатам.
        Выполняется только если требуются лицензии.
        """
        if not self.tender.is_license_production:
            return True

        WORD_MATCH_THRESHOLD = 0.7
        certificates_mentioned = False

        license_comment = self.tender.upload_license_documents_comment
        question = (
            "Какая лицензия?"
            if "лиценз" in license_comment.lower()
            else "Какой сертификат?"
        )

        answer = self.qa_model(question=question, context=license_comment)[
            "answer"
        ]

        keywords = [
            word[: len(word) - 2] for word in answer.split() if len(word) > 6
        ]
        required_matches = len(answer.split())

        for doc_idx, doc_text in enumerate(
            [self.technical_task, self.contract]
        ):
            doc_contains_certificates = False

            for sentence in doc_text:
                sentence_lower = sentence.lower()
                matching_words = sum(
                    1 for word in keywords if word in sentence_lower
                )

                if matching_words / required_matches > WORD_MATCH_THRESHOLD:
                    certificates_mentioned = True
                    doc_contains_certificates = True
                    break

            if not doc_contains_certificates:
                logger.warning(
                    f"Требования к сертификатам/лицензиям не найдены в {
                        DOCUMENT_NAMES[doc_idx]}"
                )

        return certificates_mentioned

    def validate_delivery_schedule(self) -> bool:
        """Проверяет соответствие графика поставки в документах"""
        DELIVERY_KEYWORDS = ["срок", "этап", "график"]
        DELIVERY_CLASSES = ["дата поставки", "другое"]
        PROBABILITY_THRESHOLD = 0.75

        delivery = self.tender.deliveries[0]
        schedule_mentioned = False

        if delivery.period_days_from is None:
            key_dates = [
                delivery.period_date_from[0:2],
                delivery.period_date_from[3:5],
                delivery.period_date_to[0:2],
                delivery.period_date_to[3:5],
            ]
        else:
            key_dates = [delivery.period_days_from, delivery.period_days_from]

        for doc_idx, doc_text in enumerate(
            [self.technical_task, self.contract]
        ):
            matching_sentences = []

            for sentence in doc_text:
                sentence_lower = sentence.lower()
                has_dates = any(
                    str(date) in sentence_lower for date in key_dates
                )
                has_keywords = any(
                    keyword in sentence_lower for keyword in DELIVERY_KEYWORDS
                )

                if has_dates and has_keywords:
                    matching_sentences.append(sentence.strip())

            for sentence in matching_sentences:
                probability = predict_zero_shot(
                    sentence,
                    DELIVERY_CLASSES,
                    self.zero_shot_model,
                    self.zero_shot_tokenizer,
                )[0]

                if probability > PROBABILITY_THRESHOLD:
                    schedule_mentioned = True
                    break

            if not schedule_mentioned:
                logger.warning(
                    f"График поставки не найден в {DOCUMENT_NAMES[doc_idx]}"
                )

        return schedule_mentioned

    def validate_price(self) -> bool:
        """Проверяет указание цены контракта в документах"""
        contract_price = False
        try:
            contract_price = str(round(self.tender.contract_cost))
        except:
            pass
        price_mentioned = False
        for doc_idx, doc_text in enumerate(
            [self.technical_task, self.contract]
        ):
            doc_contains_price = False

            for sentence in doc_text:
                if contract_price != False:
                    if contract_price in sentence.lower():
                        price_mentioned = True
                        doc_contains_price = True
                        break
                else:
                    if "цена контракта" in sentence.lower():
                        price_mentioned = True
                        doc_contains_price = True
                        break

            if not doc_contains_price:
                if contract_price != False:
                    logger.warning(
                        f"Максимальная цена контракта не найдена в {
                            DOCUMENT_NAMES[doc_idx]}"
                    )
                else:
                    logger.warning(
                        f"Начальная цена контракта не найдена в {
                            DOCUMENT_NAMES[doc_idx]}"
                    )

        return price_mentioned

    def validate_specifications(self) -> bool:
        """Проверяет наличие спецификаций товаров в техническом задании"""
        found_items = 0

        for item in self.tender.items:
            answer = self.qa_model(
                question="Наименование товара", context=item.name
            )["answer"]

            keywords = [word.lower() for word in answer.split()]
            item_found = False

            for sentence in self.technical_task:
                sentence_lower = sentence.lower()
                matching_words = sum(
                    1 for word in keywords if word in sentence_lower
                )

                if matching_words == len(keywords):
                    found_items += 1
                    item_found = True
                    break

            if not item_found:
                logger.warning(
                    f'Товар "{item.name}" не найден в техническом задании'
                )

        return found_items == len(self.tender.items)
