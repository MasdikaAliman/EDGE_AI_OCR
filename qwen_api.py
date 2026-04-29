import requests
from fastapi import FastAPI, HTTPException
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from PIL import Image
import base64
import json
import math
import re
import io
import logging
from typing import Union, Literal, Optional
from prompts import DOCUMENT_PROMPTS, DEFAULT_USER_PROMPTS, get_prompt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Document OCR API",
    description="API for OCR extraction from images using LLM.",
    version="2.0.0"
)

BASE_URL_LLM = "http://192.168.13.176:8053"

model = init_chat_model(
    model="qwen3-vl",
    model_provider="openai",
    base_url=BASE_URL_LLM + "/v1",
    api_key="EMPTY",
    temperature=0.0
)


class ImageUrl(BaseModel):
    """Target image URL — must be a base64 data URI (e.g. 'data:image/jpeg;base64,...'). External http/https URLs are rejected."""
    url: str = Field(
        ...,
        description="Base64-encoded image URI in the format 'data:image/<mime>;base64,<data>'. External URLs (http/https) are NOT allowed.",
        examples=["data:image/jpeg;base64,/9j/4AAQ..."]
    )

class TextContent(BaseModel):
    """A plain text message item in the content list."""
    type: Literal["text"] = Field(..., description="Content item type. Must be 'text'.")
    text: str = Field(..., description="The text instruction or prompt to send alongside the image(s).", examples=["Extract all fields from this invoice"])

class ImageContent(BaseModel):
    """An image item in the content list, wrapped in the OpenAI-compatible image_url structure."""
    type: Literal["image_url"] = Field(..., description="Content item type. Must be 'image_url'.")
    image_url: ImageUrl = Field(..., description="The image URL wrapper containing the base64 data URI.")

class MessageContent(BaseModel):
    """
    Main request body for POST /ocr/process.

    - **document_type**: Selects the specialized system prompt for the document. Defaults to 'General'. Options: General, KTP, KK, NPWP, Invoice, Quotation, SIM.
    - **fields**: Optional. If provided, the AI will extract ONLY these fields and name them exactly as listed.
    - **content**: List of text and/or image items. At least one image_url is required.

    **Max 5 images per request.**
    """
    document_type: Literal["General", "KTP", "KK", "NPWP", "Invoice", "Quotation", "SIM"] = Field(
        default="General",
        description="Document type determines which specialized extraction prompt is used automatically.\nOptions: General, KTP, KK, NPWP, Invoice, Quotation, SIM.",
        examples=["Invoice"]
    )
    fields: Optional[List[str]] = Field(
        default=None,
        description="Optional list of exact snake_case field names to extract. \nIf provided, the AI returns ONLY these keys. If omitted, the default fields for the document_type are used.",
    )
    content: List[Union[TextContent, ImageContent]] = Field(
        ...,
        description="Ordered list of content items. Include at least one image_url. Optionally prepend a text item for a custom instruction. Maximum 5 image_url items.",
        examples=[[
            {"type": "text", "text": "Extract all fields"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}
        ]]
    )



def preprocess_image(base64_str: str, max_size: int = 1024) -> str:
    with io.BytesIO(base64.b64decode(base64_str)) as src:
        img = Image.open(src)
        img.load()  # force load before BytesIO closes
    img = img.convert("RGB")

    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    with io.BytesIO() as out:
        img.save(out, format="JPEG", quality=95)
        result = base64.b64encode(out.getvalue()).decode("utf-8")

    img.close()
    return result

def sanitize_content(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Walk through message content and preprocess any images found."""
    sanitized = []
    for item in content:
        if item.get("type") == "image_url":
            raw_url: str = item["image_url"]["url"]

            # Reject external URLs — server must not download remote images
            if raw_url.startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "error_type": "url_not_allowed",
                        "message": "External image URLs are not supported. Please send images as base64 data URIs or upload files via multipart/form-data.",
                    }
                )

            # Extract base64 data (handles "data:image/...;base64,<data>")
            if ";base64," in raw_url:
                prefix, b64_data = raw_url.split(";base64,", 1)
                cleaned_b64 = preprocess_image(b64_data)
                item = {**item, "image_url": {"url": f"{prefix};base64,{cleaned_b64}"}}

        sanitized.append(item)
    return sanitized

def clean_json_response(content: str) -> str:
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    content = content.strip()
    if '{' in content:
        content = content[content.find('{'):]
    elif '[' in content:
        content = content[content.find('['):]
    if '}' in content:
        content = content[:content.rfind('}') + 1]
    elif ']' in content:
        content = content[:content.rfind(']') + 1]
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    return content


MAX_IMAGES = 5

async def process_ocr(req: MessageContent) -> dict:
    doc_type = req.document_type
    # Use get_prompt() — if user supplied fields, it injects them into the prompt
    system_prompt = get_prompt(doc_type, req.fields)
    system_message = SystemMessage(content=system_prompt)

    raw_content = [item.model_dump() for item in req.content]

    # Validate image count before sending to vLLM
    image_count = sum(1 for item in raw_content if item.get("type") == "image_url")
    if image_count > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_type": "image_limit_exceeded",
                "message": f"Too many images. Maximum allowed is {MAX_IMAGES}, but {image_count} were provided.",
            }
        )

    # If user didn't provide a text prompt, inject the default one for the document type
    has_text = any(item.get("type") == "text" for item in raw_content)
    if not has_text:
        default_prompt = DEFAULT_USER_PROMPTS.get(doc_type, DEFAULT_USER_PROMPTS["General"])
        raw_content.insert(0, {"type": "text", "text": default_prompt})

    clean_content = sanitize_content(raw_content)
    messages = [
        system_message,
        HumanMessage(content=clean_content)
    ]

    try:
        response = model.invoke(messages)
        extracted_data = json.loads(clean_json_response(response.content))
        return {"success": True, "data": extracted_data}

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_type": "json_parse_error",
                "message": "Model returned malformed JSON",
                "detail": str(e),
            }
        )

    except Exception as e:
        error_str = str(e)
        if "Error code: 400" in error_str or "BadRequestError" in error_str:
            inner_msg = error_str
            try:
                import re
                match = re.search(r"'message':\s*'([^']+)'", error_str)
                if match:
                    inner_msg = match.group(1)
            except Exception :
                pass

            logger.error(f"vLLM bad request: {inner_msg}")
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error_type": "llm_bad_request",
                    "message": inner_msg,
                    "hint": "Input is likely too long. Reduce image size or text length.",
                }
            )

        if "Error code: 401" in error_str:
            logger.error(f"vLLM auth error: {error_str}")
            raise HTTPException(
                status_code=401,
                detail={
                    "success": False,
                    "error_type": "llm_auth_error",
                    "message": "Unauthorized check your LLM API key.",
                }
            )

        if "Error code: 429" in error_str:
            logger.error(f"vLLM rate limit: {error_str}")
            raise HTTPException(
                status_code=429,
                detail={
                    "success": False,
                    "error_type": "llm_rate_limit",
                    "message": "LLM server is overloaded. Retry after a moment.",
                }
            )

        if "Error code: 503" in error_str or "Error code: 500" in error_str:
            logger.error(f"vLLM server error: {error_str}")
            raise HTTPException(
                status_code=502,
                detail={
                    "success": False,
                    "error_type": "llm_server_error",
                    "message": "LLM backend returned a server error.",
                    "detail": error_str,
                }
            )

        if "ConnectionError" in type(e).__name__ or "ConnectError" in error_str:
            logger.error(f"Cannot reach vLLM server: {error_str}")
            raise HTTPException(
                status_code=503,
                detail={
                    "success": False,
                    "error_type": "llm_unreachable",
                    "message": f"Cannot connect to LLM server at {BASE_URL_LLM}.",
                }
            )

        if "TimeoutError" in type(e).__name__ or "timed out" in error_str.lower():
            logger.error(f"vLLM request timed out: {error_str}")
            raise HTTPException(
                status_code=504,
                detail={
                    "success": False,
                    "error_type": "llm_timeout",
                    "message": "LLM server did not respond in time. Try a smaller image.",
                }
            )


        logger.error(f"Unhandled processing error [{type(e).__name__}]: {error_str}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_type": "internal_error",
                "message": f"Unexpected error: {type(e).__name__}",
                "detail": error_str,
            }
        )

@app.post("/ocr/process")
async def process_image_ocr(req: MessageContent):
    try:
        return await process_ocr(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    try:
        response = requests.get(f"{BASE_URL_LLM}/health", timeout=5)

        if response.status_code == 200:
            return {
                "status": "Healthy",
                "model_ready": True,
                "vllm_max_model_len": 11000,
                "width": 512,
                "height": 512
            }
        else:
            return {
                "status": f"Unhealthy (code: {response.status_code})",
                "model_ready": False,
                "vllm_max_model_len": 11000,
                "width": 512,
                "height": 512
            }

    except requests.exceptions.ConnectionError:
        return {
            "status": "vLLM server not reachable",
            "model_ready": False,
            "vllm_max_model_len": 11000,
            "width": 512,
            "height": 512
        }

    except requests.exceptions.Timeout:
        return {
            "status": "vLLM server timeout",
            "model_ready": False,
            "vllm_max_model_len": 11000,
            "width": 512,
            "height": 512
        }

    except Exception as e:
        return {
            "status": f"Unexpected error: {str(e)}",
            "model_ready": False,
            "vllm_max_model_len": 11000,
            "width": 512,
            "height": 512
        }


@app.get("/")
async def root():
    return {
        "name": "Document OCR API",
        "version": "2.0.0",
        "endpoints": {"image_ocr_process": "/ocr/process", "health": "/health"},
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5030)