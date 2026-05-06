import requests
from typing import Annotated, Any, Dict, List, Optional
from pydantic import WithJsonSchema
from fastapi import FastAPI, File, Form, HTTPException, UploadFile as UF
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from prompts import DOCUMENT_PROMPTS
from config import logger, MAX_IMAGES, BASE_URL_LLM, DocumentType
from utils import _file_to_content_item, _pdf_to_images
from docling_handler import _DOCLING_AVAILABLE, _extract_pages_with_docling, PageData
from ocr_service import _run_ocr, _run_langgraph_ocr

# Workaround for broken file picker in swagger
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

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
    **Multipart mode** — upload image files directly from disk.

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
                        page = all_pages[0]
                        return await _run_ocr(
                            document_type,
                            [{"type": "text", "text": prompt_text}, page["image"]]
                            + page.get("table_images", []),
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
