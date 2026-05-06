import base64
import io
import json
import logging
import re
import tempfile
import os
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile as UF
from typing import Annotated
from pydantic import WithJsonSchema
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
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

# Workaround for broken file picker in swagger
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]



DocumentType = Literal["General", "KTP", "KK", "NPWP", "Invoice", "Quotation", "SIM", "STNK", "Passport"]

MAX_IMAGES = 5
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/tiff", "application/pdf"}

# ── Docling availability check ─────────────────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, EasyOcrOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.datamodel.base_models import InputFormat
    _DOCLING_AVAILABLE = True
    logger.info("Docling is available  PDF text extraction enabled.")
except ImportError:
    _DOCLING_AVAILABLE = False
    logger.warning("Docling not installed. PDFs will fall back to image-based OCR.")


app = FastAPI(
    title="Document OCR API",
    description=(
        "High-precision document OCR powered by Qwen3-VL via vLLM.\n\n"
        "Accepts images as **base64 JSON** or **multipart file uploads**.\n\n"
        "Features a **LangGraph-powered page-by-page pipeline** that sends each page's "
        "full image, table crops, and extracted markdown to the VLM for maximum accuracy.\n\n"
        "PDFs are processed via **Docling** (page images + table crops + markdown) when available, "
        "falling back to pdfplumber image-based OCR.\n\n"
        "Supported document types: "
        + ", ".join(DOCUMENT_PROMPTS.keys())
    ),
    version="5.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi():
    """Serve OpenAPI schema with explicit UTF-8 charset to fix Swagger UI encoding on remote access."""
    return JSONResponse(
        content=app.openapi(),
        media_type="application/json; charset=utf-8",
    )


BASE_URL_LLM = "http://192.168.13.176:8053"
# BASE_URL_LLM = "http://localhost:1234"

model = init_chat_model(
    model="qwen3-vl",
    model_provider="openai",
    base_url=BASE_URL_LLM + "/v1",
    api_key="EMPTY",
    temperature=0.0,
)


# ── Image helpers ──────────────────────────────────────────────────────────────

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


# ── Docling PDF extraction (page-by-page) ──────────────────────────────────────

class PageData(TypedDict):
    page_no: int
    image: Dict[str, Any]               # Full page base64 image_url content dict
    table_images: List[Dict[str, Any]]   # Table crop base64 image_url content dicts
    markdown: str                        # Extracted text for this page


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


def _extract_pages_with_docling(pdf_bytes: bytes) -> List[PageData]:
    """
    Use Docling to extract per-page data from a PDF:
      - full page image (base64)
      - table crop images (base64)
      - page-specific markdown text

    Returns a list of PageData dicts, one per page.
    """
    if not _DOCLING_AVAILABLE:
        raise RuntimeError("Docling is not installed.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
        )
        pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CPU,
        )
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

        result = converter.convert(tmp_path)
        doc = result.document
        pages: List[PageData] = []

        for page_no, page in doc.pages.items():
            # 1. Full page image
            if page.image is None:
                logger.warning("Page %d has no image, skipping.", page_no)
                continue

            page_image_item = _pil_to_content_item(page.image.pil_image)

            # 2. Table images for this page
            table_image_items: List[Dict[str, Any]] = []
            for table in doc.tables:
                is_on_page = False
                if hasattr(table, "prov") and table.prov:
                    for p in table.prov:
                        if p.page_no == page_no:
                            is_on_page = True
                            break
                if is_on_page:
                    try:
                        table_img = table.get_image(doc)
                        if table_img:
                            table_image_items.append(_pil_to_content_item(table_img))
                    except Exception as e:
                        logger.warning("Could not extract table image on page %d: %s", page_no, e)

            # 3. Page-specific text
            page_text = ""
            for item, _level in doc.iterate_items():
                if hasattr(item, "prov") and item.prov:
                    if any(p.page_no == page_no for p in item.prov):
                        if hasattr(item, "text"):
                            page_text += item.text + "\n"

            pages.append(PageData(
                page_no=page_no,
                image=page_image_item,
                table_images=table_image_items,
                markdown=page_text,
            ))

        logger.info("Docling extracted %d pages from PDF.", len(pages))
        return pages

    finally:
        os.unlink(tmp_path)


# ── Single-image OCR (unchanged) ───────────────────────────────────────────────

async def _run_ocr(
    document_type: str,
    raw_content: List[Dict[str, Any]],
    fields: Optional[List[str]] = None,
) -> dict:
    """
    Shared processing logic for a single image.

    Args:
        document_type: One of the keys in DOCUMENT_PROMPTS.
        raw_content:   List of plain dicts with 'type' == 'text' or 'image_url'.
        fields:        Optional list of field names to override defaults.

    Returns:
        {"success": True, "data": <extracted dict>}

    Raises:
        HTTPException on all known failure modes.
    """
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


# ── LangGraph State ────────────────────────────────────────────────────────────

class OCRState(TypedDict):
    document_type: str
    fields: Optional[List[str]]
    custom_prompt: Optional[str]

    # Unified page list (works for both Docling PDFs and plain images)
    pages: List[PageData]
    current_idx: int

    # Results
    page_results: List[Dict[str, Any]]
    final_result: Dict[str, Any]


# ── Unified LangGraph nodes ───────────────────────────────────────────────────

def process_page_node(state: OCRState) -> Dict[str, Any]:
    """
    Process a single page by sending the LLM:
      1. The page markdown (as text)
      2. The full page image
      3. Any table crop images for that page
    """
    idx = state["current_idx"]
    page_data = state["pages"][idx]
    doc_type = state["document_type"]
    fields = state["fields"]
    custom_prompt = state.get("custom_prompt", "")

    user_text = custom_prompt or DEFAULT_USER_PROMPTS.get(doc_type, DEFAULT_USER_PROMPTS["General"])

    # Build multimodal content list
    content: List[Dict[str, Any]] = []
    page_markdown = page_data.get("markdown", "")
    if page_markdown:
        content.append({
            "type": "text",
            "text": f"{user_text}\n\n### Page {page_data['page_no']} Text/Markdown:\n{page_markdown}",
        })
    else:
        content.append({"type": "text", "text": user_text})

    content.append(page_data["image"])
    for table_img in page_data.get("table_images", []):
        content.append(table_img)

    system_prompt = get_prompt(doc_type, fields)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]

    try:
        response = model.invoke(messages)
        extracted_data = json.loads(_clean_json_response(response.content))
        logger.info("Page %d extracted by image successfully.", page_data["page_no"])
    except Exception as e:
        logger.error("Error extracting page %d: %s", page_data["page_no"], e)
        extracted_data = {"error": str(e), "page": page_data["page_no"]}

    return {
        "page_results": state.get("page_results", []) + [extracted_data],
        "current_idx": idx + 1,
    }


def reprocess_page_node(state: OCRState) -> Dict[str, Any]:
    """Re-extract missing fields from the last processed page."""
    idx = state["current_idx"] - 1
    page_data = state["pages"][idx]
    required_fields = state.get("fields") or []
    last_result = state["page_results"][-1]

    missing = [
        f for f in required_fields
        if f not in last_result or last_result.get(f) in ["", None]
    ]
    if not missing:
        return {}

    prompt = f"""Some fields were missing from the previous extraction:
{missing}

Re-analyze the document carefully, focusing on tables and structured rows.
IMPORTANT:
- Extract ONLY the missing fields listed above.
- Do NOT overwrite fields that were already correctly extracted.
- Return ONLY a raw JSON object. No markdown, no commentary.
"""
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append(page_data["image"])
    for table_img in page_data.get("table_images", []):
        content.append(table_img)

    messages = [
        SystemMessage(content="You are an expert at extracting structured data from documents."),
        HumanMessage(content=content),
    ]

    try:
        response = model.invoke(messages)
        new_data = json.loads(_clean_json_response(response.content))
        logger.info("Reprocess page %d found additional data.", page_data["page_no"])
    except Exception:
        new_data = {}

    merged = {**last_result, **new_data}
    updated_results = state["page_results"][:-1] + [merged]
    return {"page_results": updated_results}


# ── Routing & aggregation ─────────────────────────────────────────────────────


def check_missing_fields(state: OCRState) -> str:
    """After extraction, check whether any required fields are still missing."""
    last_result = state["page_results"][-1] if state["page_results"] else {}

    # Filter out blank/None entries  FastAPI can pass [""] for an empty form field
    required_fields = [
        f for f in (state.get("fields") or [])
        if f and f.strip()
    ]

    if not required_fields:
        return "next_step"

    missing = [
        f for f in required_fields
        if f not in last_result or last_result.get(f) in ["", None]
    ]

    if missing:
        logger.info("Missing fields detected, will reprocess: %s", missing)
        return "reprocess_page"

    return "next_step"


def next_step(state: OCRState) -> str:
    """Decide whether to loop back for the next page or move to aggregation."""
    if state["current_idx"] < len(state["pages"]):
        return "process_page"
    return "aggregate_results"


def aggregate_node(state: OCRState) -> Dict[str, Any]:
    """Merge all page results into a single JSON using the LLM."""
    page_results = state.get("page_results", [])
    doc_type = state["document_type"]

    if not page_results:
        return {"final_result": {}}

    if len(page_results) == 1:
        return {"final_result": page_results[0]}

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
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    try:
        response = model.invoke(messages)
        final_data = json.loads(_clean_json_response(response.content))
        logger.info("Aggregation complete.")
    except Exception as e:
        logger.error("Error in aggregation node: %s", e)
        final_data = {"error": f"Aggregation failed: {str(e)}", "partial_results": page_results}

    return {"final_result": final_data}


# ── LangGraph builder ──────────────────────────────────────────────────────────

async def _run_langgraph_ocr(
    document_type: str,
    pages: List[PageData],
    fields: Optional[List[str]] = None,
    custom_prompt: str = "",
) -> dict:
    workflow = StateGraph(OCRState)

    workflow.add_node("process_page",      process_page_node)
    workflow.add_node("reprocess_page",    reprocess_page_node)
    workflow.add_node("next_step",         lambda state: {})
    workflow.add_node("aggregate_results", aggregate_node)

    workflow.add_edge(START, "process_page")

    workflow.add_conditional_edges(
        "process_page",
        check_missing_fields,
        {"reprocess_page": "reprocess_page", "next_step": "next_step"},
    )
    workflow.add_edge("reprocess_page", "next_step")
    workflow.add_conditional_edges(
        "next_step",
        next_step,
        {"process_page": "process_page", "aggregate_results": "aggregate_results"},
    )
    workflow.add_edge("aggregate_results", END)

    app_graph = workflow.compile()

    initial_state: OCRState = {
        "document_type":  document_type,
        "fields":         fields,
        "custom_prompt":  custom_prompt,
        "pages":          pages,
        "current_idx":    0,
        "page_results":   [],
        "final_result":   {},
    }

    try:
        final_state = await app_graph.ainvoke(initial_state)
        return {"success": True, "data": final_state.get("final_result", {})}
    except Exception as e:
        logger.error("LangGraph processing error: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error_type": "graph_error", "message": str(e)},
        )


# ── Error handler ──────────────────────────────────────────────────────────────

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
            "Unauthorized  check your LLM API key.",
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


# ── API endpoint ───────────────────────────────────────────────────────────────

@app.post(
    "/ocr/process/upload",
    summary="Extract document fields (multipart file upload)",
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
        example=None,
    ),
    custom_prompt: Optional[str] = Form(
        default="",
        description=(
            "Optional custom instruction prepended to the model input. "
            "If omitted, the default instruction for the document_type is used."
        ),
    ),
    use_docling: bool = Form(
        default=True,
        description=(
            "If True (default) and Docling is installed, native PDFs are parsed "
            "via Docling (markdown + tables) instead of rendering pages to images. "
            "Set to False to force the image-based VLM path for all inputs."
        ),
    ),
):
    """
    **Multipart mode**  upload image files directly from disk.

    Send a `multipart/form-data` POST with:
    - `files`: one or more image or PDF files
    - `document_type`: e.g. `Invoice` (default: `General`)
    - `fields` *(optional)*: list of field names to extract
    - `custom_prompt` *(optional)*: override the default user instruction
    - `use_docling` *(optional)*: set `false` to force image-based OCR for PDFs

    **PDF routing logic:**
    1. If `use_docling=true` and Docling is installed → Docling markdown path
    2. Otherwise → pdfplumber image rendering → VLM image path

    ```bash
    curl -X 'POST' \\
        'http://localhost:5030/ocr/process/upload' \\
        -H 'accept: application/json' \\
        -H 'Content-Type: multipart/form-data' \\
        -F 'files=@invoice.pdf;type=application/pdf' \\
        -F 'document_type=Invoice' \\
        -F 'use_docling=true'
    ```
    """
    # Strip blank entries that FastAPI may inject when the form field is submitted empty
    parsed_fields = [f for f in (fields or []) if f and f.strip()] or None
    pdf_bytes_store: Optional[bytes] = None
    image_pages: List[PageData] = []
    prompt_text = custom_prompt or ""

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
                pdf_bytes_store = raw_bytes
            else:
                content_item = _file_to_content_item(raw_bytes, content_type)
                image_pages.append(PageData(
                    page_no=len(image_pages) + 1,
                    image=content_item,
                    table_images=[],
                    markdown="",
                ))

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
        # ── Docling path (PDF) ─────────────────────────────────────────────────
        if pdf_bytes_store is not None:
            if use_docling and _DOCLING_AVAILABLE:
                logger.info("Routing PDF through Docling page-by-page pipeline.")
                try:
                    docling_pages = _extract_pages_with_docling(pdf_bytes_store)
                except Exception as e:
                    logger.warning("Docling failed (%s), falling back to image OCR.", e)
                    docling_pages = None

                if docling_pages:
                    all_pages = docling_pages + image_pages
                    if len(all_pages) == 1:
                        return await _run_ocr(
                            document_type,
                            [{"type": "text", "text": prompt_text}, all_pages[0]["image"]]
                            + all_pages[0].get("table_images", []),
                            parsed_fields,
                        )
                    return await _run_langgraph_ocr(
                        document_type, all_pages, parsed_fields, prompt_text,
                    )

            # Fallback: render PDF pages to images via pdfplumber
            logger.info("Falling back to pdfplumber image-based OCR.")
            pdf_image_items = _pdf_to_images(pdf_bytes_store)
            for i, img_item in enumerate(pdf_image_items):
                image_pages.append(PageData(
                    page_no=len(image_pages) + 1,
                    image=img_item,
                    table_images=[],
                    markdown="",
                ))

        # ── Validate we have at least one page ─────────────────────────────────
        if not image_pages:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error_type": "no_image_provided",
                    "message": "At least one image or PDF is required.",
                },
            )

        if len(image_pages) > MAX_IMAGES:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error_type": "image_limit_exceeded",
                    "message": f"Too many pages. Maximum is {MAX_IMAGES}, but {len(image_pages)} were provided.",
                },
            )

        # ── Single page → direct OCR; multi-page → LangGraph ──────────────────
        if len(image_pages) == 1:
            page = image_pages[0]
            raw_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
            raw_content.append(page["image"])
            raw_content.extend(page.get("table_images", []))
            return await _run_ocr(document_type, raw_content, parsed_fields)

        return await _run_langgraph_ocr(
            document_type, image_pages, parsed_fields, prompt_text,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ── Health & root ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
async def health_check():
    """Check connectivity to the vLLM backend."""
    base_info = {
        "vllm_max_model_len": 11000,
        "max_image_size": "1024×1024",
        "max_images_per_request": MAX_IMAGES,
        "docling_available": _DOCLING_AVAILABLE,
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
        "version": "5.0.0",
        "docling_available": _DOCLING_AVAILABLE,
        "supported_document_types": list(DOCUMENT_PROMPTS.keys()),
        "endpoints": {
            "upload_ocr": "POST /ocr/process/upload",
            "health":     "GET  /health",
            "docs":       "GET  /docs",
            "redoc":      "GET  /redoc",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5030)