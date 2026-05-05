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
import pdfplumber

from prompts import DEFAULT_USER_PROMPTS, DOCUMENT_PROMPTS, get_prompt
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Union, TypedDict
from langgraph.graph import StateGraph, START, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DocumentType = Literal["General", "KTP", "KK", "NPWP", "Invoice", "Quotation", "SIM", "STNK", "Passport"]

MAX_IMAGES = 5
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/tiff", "application/pdf"}


app = FastAPI(
    title="Document OCR API",
    description=(
        "High-precision document OCR powered by Qwen3-VL via vLLM.\n\n"
        "Accepts images as **base64 JSON** or **multipart file uploads**.\n\n"
        "Features a **LangGraph-powered pipeline** for multi-page documents and PDFs.\n\n"
        "Supported document types: "
        + ", ".join(DOCUMENT_PROMPTS.keys())
    ),
    version="3.1.0",
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


def _pdf_to_images(pdf_bytes: bytes, max_pages: int = MAX_IMAGES) -> List[Dict[str, Any]]:
    """Convert a PDF file into a list of image_url content items."""
    items = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                # if i >= max_pages:
                #     break
                
                # Render page to image at 150 DPI
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
        logger.error("Error processing PDF: %s", e)
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


# ── LangGraph Pipeline for Multi-Page ──────────────────────────────────────

class OCRState(TypedDict):
    document_type: str
    fields: Optional[List[str]]
    custom_prompt: Optional[str]
    images: List[Dict[str, Any]]
    current_idx: int
    page_results: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    missing_fields: List[str]


def process_page_node(state: OCRState) -> Dict[str, Any]:
    idx = state["current_idx"]
    image_item = state["images"][idx]
    
    doc_type = state["document_type"]
    fields = state["fields"]
    custom_prompt = state["custom_prompt"]
    
    raw_content = [image_item]
    if custom_prompt:
        raw_content.insert(0, {"type": "text", "text": custom_prompt})
    else:
        default_prompt = DEFAULT_USER_PROMPTS.get(doc_type, DEFAULT_USER_PROMPTS["General"])
        raw_content.insert(0, {"type": "text", "text": default_prompt})
        
    system_prompt = get_prompt(doc_type, fields)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=raw_content),
    ]
    
    try:
        response = model.invoke(messages)
        extracted_data = json.loads(_clean_json_response(response.content))
        print(f"\n=== DEBUG: Page {idx} Extracted Data ===\n{response.content}")
    except Exception as e:
        logger.error("Error extracting page %d: %s", idx, e)
        extracted_data = {"error": str(e), "page": idx}
        
    return {
        "page_results": state.get("page_results", []) + [extracted_data],
        "current_idx": idx + 1
    }

def check_more_pages(state: OCRState) -> Literal["process_page", "aggregate_results"]:
    if state["current_idx"] < len(state["images"]):
        return "process_page"
    return "aggregate_results"

def check_missing_fields(state: OCRState) -> str:
    last_result = state["page_results"][-1] if state["page_results"] else {}
    required_fields = state.get("fields") or []

    if not required_fields:
        return "next_step"

    missing = [
        f for f in required_fields
        if f not in last_result or last_result.get(f) in ["", None]
    ]

    if missing:
        return "reprocess_page"

    return "next_step"

def reprocess_page_node(state: OCRState) -> Dict[str, Any]:
    idx = state["current_idx"] - 1  # last processed page
    image_item = state["images"][idx]

    required_fields = state.get("fields") or []
    last_result = state["page_results"][-1]

    missing = [
        f for f in required_fields
        if f not in last_result or last_result.get(f) in ["", None]
    ]

    if not missing:
        return {}

    prompt = f"""
Some fields were missing from previous extraction:
{missing}

Re-analyze the image carefully.

IMPORTANT:
- Focus on tables, rows, structured data
- Extract ONLY missing fields
- Do NOT overwrite existing correct fields

Return JSON only.
"""

    messages = [
        SystemMessage(content="You are an expert at extracting data from tables in documents."),
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            image_item
        ])
    ]

    try:
        response = model.invoke(messages)
        new_data = json.loads(_clean_json_response(response.content))
        print(f"DEBUG page {idx} reprocess: {response.content}")
    except Exception:
        new_data = {}

    merged = {**last_result, **new_data}
    state["page_results"][-1] = merged

    return {"page_results": state["page_results"]}

def next_step(state: OCRState) -> str:
    if state["current_idx"] < len(state["images"]):
        return "process_page"
    return "aggregate_results"

def aggregate_node(state: OCRState) -> Dict[str, Any]:
    page_results = state.get("page_results", [])
    doc_type = state["document_type"]
    
    if not page_results:
        return {"final_result": {}}
        
    system_prompt = (
    f"""You are an expert data arbitration engine for {doc_type} documents.
Multiple pages were OCR'd independently. Produce ONE final JSON object.
RULES:
- For each field, choose the most complete and credible value across all pages.
- If the same field appears on multiple pages with DIFFERENT values, prefer:
    1. The value that is more complete (not empty/partial).
    2. The value from the page where that field would naturally appear
       (e.g. totals from the last page, header info from the first page).
- For array fields (e.g. line items, members), MERGE arrays from all pages
  and deduplicate identical rows.
- If a field is empty ("") or null on ALL pages, output "" for that field.
- NEVER fabricate or infer values not present in any page result.
- Return ONLY a raw JSON object. No markdown, no commentary, no code fences.
"""
    )
    
    user_content = json.dumps(page_results, indent=2)
    print("\n=== DEBUG: Aggregation User Content ===\n", user_content)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    
    try:
        response = model.invoke(messages)
        final_data = json.loads(_clean_json_response(response.content))
        print("""\n=== DEBUG: Aggregation Node ===\n""", response.content)
    except Exception as e:
        logger.error("Error in aggregation node: %s", e)
        final_data = {"error": f"Aggregation failed: {str(e)}", "partial_results": page_results}
        
    return {"final_result": final_data}

async def _run_langgraph_ocr(
    document_type: str,
    images: List[Dict[str, Any]],
    fields: Optional[List[str]] = None,
    custom_prompt: str = ""
) -> dict:
    workflow = StateGraph(OCRState)

    workflow.add_node("process_page", process_page_node)
    workflow.add_node("reprocess_page", reprocess_page_node)
    workflow.add_node("next_step", lambda state: {})  # dummy node
    workflow.add_node("aggregate_results", aggregate_node)

    # START
    workflow.add_edge(START, "process_page")

    # AFTER process → check missing
    workflow.add_conditional_edges(
        "process_page",
        check_missing_fields,
        {
            "reprocess_page": "reprocess_page",
            "next_step": "next_step"
        }
    )

    # AFTER retry → go to next decision
    workflow.add_edge("reprocess_page", "next_step")

    # NEXT STEP routing
    workflow.add_conditional_edges(
        "next_step",
        next_step,
        {
            "process_page": "process_page",
            "aggregate_results": "aggregate_results"
        }
    )

    # END
    workflow.add_edge("aggregate_results", END)
        
    app_graph = workflow.compile()
    
    initial_state = {
        "document_type": document_type,
        "fields": fields,
        "custom_prompt": custom_prompt,
        "images": images,
        "current_idx": 0,
        "page_results": [],
        "final_result": {}
    }
    
    try:
        final_state = await app_graph.ainvoke(initial_state)
        return {"success": True, "data": final_state.get("final_result", {})}
    except Exception as e:
        logger.error("LangGraph processing error: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error_type": "graph_error", "message": str(e)}
        )


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
            f"One or more image/PDF files to process (max {MAX_IMAGES} pages/images). "
            "Accepted formats: JPEG, PNG, WebP, GIF, TIFF, PDF."
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
    mime = None
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
            mime = content_type.split(";")[0].strip().lower()

            if mime == "application/pdf":
                current_image_count = sum(1 for item in raw_content if item.get("type") == "image_url")
                remaining_slots = MAX_IMAGES - current_image_count
                
                if remaining_slots > 0:
                    pdf_items = _pdf_to_images(raw_bytes, max_pages=remaining_slots)
                    raw_content.extend(pdf_items)
            else:
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
        # Separate images and sanitize
        images = [item for item in raw_content if item.get("type") == "image_url"]
        images = _sanitize_content(images)
        
        # Extract custom prompt text if present
        prompt_text = ""
        for item in raw_content:
            if item.get("type") == "text":
                prompt_text = item.get("text", "")
                break
                
        if len(images) > 1 or mime == "application/pdf":
            return await _run_langgraph_ocr(document_type, images, parsed_fields, prompt_text)
        else:
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
        "version": "3.1.0",
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