import base64
import io
import json
import re
from typing import Any, Dict, List
from fastapi import HTTPException
from PIL import Image
import pdfplumber
from config import logger, BASE_URL_LLM, ALLOWED_MIME_TYPES, MAX_IMAGES


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


def _pil_to_content_item(pil_img: "Image.Image", mime: str = "image/png") -> Dict[str, Any]:
    """Convert a PIL Image into an OpenAI-compatible image_url content dict."""
    with io.BytesIO() as buf:
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    cleaned = _preprocess_image(b64)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{cleaned}"},
    }


def _pdf_to_images(pdf_bytes: bytes, max_pages: int = MAX_IMAGES) -> List[Dict[str, Any]]:
    """Convert a PDF file into a list of image_url content items (fallback path)."""
    items = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                im = page.to_image(resolution=500).original

                with io.BytesIO() as out:
                    if im.mode in ("RGBA", "P"):
                        im = im.convert("RGB")
                    im.save(out, format="PNG", quality=95)
                    b64 = base64.b64encode(out.getvalue()).decode("utf-8")

                cleaned = _preprocess_image(b64)
                items.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{cleaned}"},
                })
    except Exception as e:
        logger.error("Error processing PDF to images: %s", e)
        raise ValueError(f"Failed to process PDF: {str(e)}")

    return items


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

    if "{" in content:
        content = content[content.find("{"):]
    elif "[" in content:
        content = content[content.find("["):]

    if "}" in content:
        content = content[: content.rfind("}") + 1]
    elif "]" in content:
        content = content[: content.rfind("]") + 1]

    content = re.sub(r",(\s*[}\]])", r"\1", content)
    return content


def _clean_markdown_response(content: str) -> str:
    if "```markdown" in content:
        content = content.split("```markdown")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return content.strip()


def _extract_inner_message(error_str: str) -> str:
    """Try to pull the human-readable message out of a vLLM 400 error string."""
    try:
        match = re.search(r"'message':\s*'([^']+)'", error_str)
        if match:
            return match.group(1)
    except Exception:
        pass
    return error_str


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
