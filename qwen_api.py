import base64
import io
import json
import logging
import re
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image


from prompts import DEFAULT_USER_PROMPTS, DOCUMENT_PROMPTS, get_prompt
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DocumentType = Literal["General", "KTP", "KK", "NPWP", "Invoice", "Quotation", "SIM", "STNK", "Passport"]

MAX_IMAGES = 5
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/tiff"}


app = FastAPI(
    title="Document OCR API",
    description=(
        "High-precision document OCR powered by Qwen3-VL via vLLM.\n\n"
        "Accepts images as **base64 JSON** or **multipart file uploads**.\n\n"
        "Supported document types: "
        + ", ".join(DOCUMENT_PROMPTS.keys())
    ),
    version="3.0.0",
)

BASE_URL_LLM = "http://192.168.13.176:8053"

model = init_chat_model(
    model="qwen3-vl",
    model_provider="openai",
    base_url=BASE_URL_LLM + "/v1",
    api_key="EMPTY",
    temperature=0.0,
)


def _preprocess_image(b64_data: str, max_size: int = 1024) -> str:
    """Decode → resize if needed → re-encode as JPEG base64."""
    raw_bytes = base64.b64decode(b64_data)
    with io.BytesIO(raw_bytes) as src:
        img = Image.open(src)
        img.load()

    img = img.convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    with io.BytesIO() as out:
        img.save(out, format="JPEG", quality=95)
        result = base64.b64encode(out.getvalue()).decode("utf-8")

    img.close()
    return result


def _file_to_content_item(raw_bytes: bytes, content_type: str) -> Dict[str, Any]:
    """Convert raw image bytes into an OpenAI-compatible image_url content dict."""
    mime = content_type.split(";")[0].strip().lower()
    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "success": False,
                "error_type": "unsupported_media_type",
                "message": (
                    f"File type '{mime}' is not supported. "
                    f"Accepted types: {', '.join(sorted(ALLOWED_MIME_TYPES))}."
                ),
            },
        )
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    cleaned = _preprocess_image(b64)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{cleaned}"},
    }


def _sanitize_content(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized = []
    for item in content:
        if item.get("type") == "image_url":
            raw_url: str = item["image_url"]["url"]

            if raw_url.startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "error_type": "url_not_allowed",
                        "message": (
                            "External image URLs are not supported. "
                            "Send images as base64 data URIs or upload via multipart/form-data."
                        ),
                    },
                )

            if ";base64," in raw_url:
                prefix, b64_data = raw_url.split(";base64,", 1)
                cleaned = _preprocess_image(b64_data)
                item = {**item, "image_url": {"url": f"{prefix};base64,{cleaned}"}}

        sanitized.append(item)
    return sanitized


def _clean_json_response(content: str) -> str:
    """Strip markdown fences and extract the first JSON object/array."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    content = content.strip()

    # Advance to first JSON character
    if "{" in content:
        content = content[content.find("{"):]
    elif "[" in content:
        content = content[content.find("["):]

    # Trim trailing garbage after closing bracket
    if "}" in content:
        content = content[: content.rfind("}") + 1]
    elif "]" in content:
        content = content[: content.rfind("]") + 1]

    # Remove trailing commas before } or ]
    content = re.sub(r",(\s*[}\]])", r"\1", content)
    return content



async def _run_ocr(
    document_type: str,
    raw_content: List[Dict[str, Any]],
    fields: Optional[List[str]] = None,
) -> dict:
    """
    Shared processing logic.

    Args:
        document_type: One of the keys in DOCUMENT_PROMPTS.
        raw_content:   List of plain dicts with 'type' == 'text' or 'image_url'.
        fields:        Optional list of field names to override defaults.

    Returns:
        {"success": True, "data": <extracted dict>}

    Raises:
        HTTPException on all known failure modes.
    """
    # ── Validate image count ───────────────────────────────────────────────────
    image_count = sum(1 for item in raw_content if item.get("type") == "image_url")

    if image_count == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_type": "no_image_provided",
                "message": "At least one image is required.",
            },
        )

    if image_count > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_type": "image_limit_exceeded",
                "message": (
                    f"Too many images. Maximum is {MAX_IMAGES}, "
                    f"but {image_count} were provided."
                ),
            },
        )

    has_text = any(item.get("type") == "text" for item in raw_content)
    if not has_text:
        default_prompt = DEFAULT_USER_PROMPTS.get(document_type, DEFAULT_USER_PROMPTS["General"])
        raw_content = [{"type": "text", "text": default_prompt}] + raw_content

    clean_content = _sanitize_content(raw_content)
    system_prompt = get_prompt(document_type, fields)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=clean_content),
    ]
    try:
        response = model.invoke(messages, timeout=120)
        extracted_data = json.loads(_clean_json_response(response.content))
        return {"success": True, "data": extracted_data}

    except json.JSONDecodeError as e:
        logger.error("JSON parsing error: %s", e)
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error_type": "json_parse_error",
                "message": "Model returned malformed JSON.",
                "detail": str(e),
            },
        )

    except Exception as e:
        _handle_llm_exception(e)


def _handle_llm_exception(exc: Exception) -> None:
    err = str(exc)

    checks = [
        (
            "Error code: 400" in err or "BadRequestError" in err,
            400,
            "llm_bad_request",
            _extract_inner_message(err),
            "Input is likely too long. Reduce image size or text length.",
        ),
        (
            "Error code: 401" in err,
            401,
            "llm_auth_error",
            "Unauthorized — check your LLM API key.",
            None,
        ),
        (
            "Error code: 429" in err,
            429,
            "llm_rate_limit",
            "LLM server is overloaded. Retry after a moment.",
            None,
        ),
        (
            "Error code: 503" in err or "Error code: 500" in err,
            502,
            "llm_server_error",
            "LLM backend returned a server error.",
            None,
        ),
        (
            "ConnectionError" in type(exc).__name__ or "ConnectError" in err,
            503,
            "llm_unreachable",
            f"Cannot connect to LLM server at {BASE_URL_LLM}.",
            None,
        ),
        (
            "TimeoutError" in type(exc).__name__ or "timed out" in err.lower(),
            504,
            "llm_timeout",
            "LLM server did not respond in time. Try a smaller image.",
            None,
        ),
    ]

    for condition, status, error_type, message, hint in checks:
        if condition:
            detail: Dict[str, Any] = {
                "success": False,
                "error_type": error_type,
                "message": message,
            }
            if hint:
                detail["hint"] = hint
            logger.error("vLLM error [%s]: %s", error_type, message)
            raise HTTPException(status_code=status, detail=detail)


    logger.error("Unhandled processing error [%s]: %s", type(exc).__name__, err)
    raise HTTPException(
        status_code=500,
        detail={
            "success": False,
            "error_type": "internal_error",
            "message": f"Unexpected error: {type(exc).__name__}",
            "detail": err,
        },
    )


def _extract_inner_message(error_str: str) -> str:
    """Try to pull the human-readable message out of a vLLM 400 error string."""
    try:
        match = re.search(r"'message':\s*'([^']+)'", error_str)
        if match:
            return match.group(1)
    except Exception:
        pass
    return error_str


@app.post(
    "/ocr/process/upload",
    summary="Extract document fields (multipart — file upload)",
    tags=["OCR"],
)
async def process_ocr_upload(
    files: List[UploadFile] = File(
        ...,
        description=(
            f"One or more image files to process (max {MAX_IMAGES}). "
            "Accepted formats: JPEG, PNG, WebP, GIF, TIFF."
        ),
    ),
    document_type: DocumentType = Form(
        default="General",
        description="Document type. Options: " + ", ".join(DOCUMENT_PROMPTS.keys()),
    ),
    fields: Optional[List[str]] = Form(
        default=[],
        description=(
            "Optional list of snake_case field names to extract. "
            "Add each field name as a separate 'fields' parameter."
        ),
        example=None
    ),
    custom_prompt: Optional[str] = Form(
        default="",
        description=(
            "Optional custom instruction prepended to the model input. "
            "If omitted, the default instruction for the document_type is used."
        ),
    ),
):
    """
    **Multipart mode** — upload image files directly from disk.

    Send a `multipart/form-data` POST with:
    - `files`: one or more image files
    - `document_type`: e.g. `KTP` (default: `General`)
    - `fields` *(optional)*: List of field names, e.g., pass multiple `-F "fields=name,nik"`
    - `custom_prompt` *(optional)*: override the default user instruction

    ```bash
    curl -X 'POST' \
        'http://localhost:5030/ocr/process/upload' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F 'files=@ktp_2.png;type=image/png' \
        -F 'document_type=KTP' \
        -F 'fields=name,nik' \
        -F 'custom_prompt=string'
    ```
    """
    parsed_fields = fields

    # ── Build content list from uploaded files ─────────────────────────────────
    raw_content: List[Dict[str, Any]] = []

    if custom_prompt:
        raw_content.append({"type": "text", "text": custom_prompt})

    for upload in files:
        content_type = upload.content_type or "application/octet-stream"
        raw_bytes = await upload.read()

        if not raw_bytes:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error_type": "empty_file",
                    "message": f"Uploaded file '{upload.filename}' is empty.",
                },
            )

        try:
            raw_content.append(_file_to_content_item(raw_bytes, content_type))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error_type": "file_read_error",
                    "message": f"Could not process file '{upload.filename}': {e}",
                },
            )

    try:
        return await _run_ocr(document_type, raw_content, parsed_fields)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/health", tags=["Utility"])
async def health_check():
    """Check connectivity to the vLLM backend."""
    base_info = {
        "vllm_max_model_len": 11000,
        "max_image_size": "1024×1024",
        "max_images_per_request": MAX_IMAGES,
    }
    try:
        resp = requests.get(f"{BASE_URL_LLM}/health", timeout=5)
        if resp.status_code == 200:
            return {"status": "Healthy", "model_ready": True, **base_info}
        return {
            "status": f"Unhealthy (HTTP {resp.status_code})",
            "model_ready": False,
            **base_info,
        }
    except requests.exceptions.ConnectionError:
        return {"status": "vLLM server not reachable", "model_ready": False, **base_info}
    except requests.exceptions.Timeout:
        return {"status": "vLLM server timeout", "model_ready": False, **base_info}
    except Exception as e:
        return {"status": f"Unexpected error: {e}", "model_ready": False, **base_info}


@app.get("/", tags=["Utility"])
async def root():
    return {
        "name": "Document OCR API",
        "version": "3.0.0",
        "supported_document_types": list(DOCUMENT_PROMPTS.keys()),
        "endpoints": {
            "json_ocr":    "POST /ocr/process",
            "upload_ocr":  "POST /ocr/process/upload",
            "health":      "GET  /health",
            "docs":        "GET  /docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5030)