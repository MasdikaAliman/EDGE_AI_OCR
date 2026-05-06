import logging
from typing import Literal
from langchain.chat_models import init_chat_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_IMAGES = 5
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/tiff", "application/pdf"}

# BASE_URL_LLM = "http://192.168.13.176:8053"
BASE_URL_LLM = "http://localhost:1234"

DocumentType = Literal["General", "KTP", "KK", "NPWP", "Invoice", "Quotation", "SIM", "STNK", "Passport"]

model = init_chat_model(
    # model="qwen3-vl",
    model="qwen/qwen3-vl-4b",
    model_provider="openai",
    base_url=BASE_URL_LLM + "/v1",
    api_key="EMPTY",
    temperature=0.0,
)
