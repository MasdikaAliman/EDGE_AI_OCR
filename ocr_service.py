import json
from typing import Any, Dict, List, Optional, TypedDict
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from prompts import DEFAULT_USER_PROMPTS, get_prompt
from config import logger, MAX_IMAGES, model
from utils import _sanitize_content, _clean_json_response, _handle_llm_exception
from docling_handler import PageData


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


# ── Single-image OCR ──────────────────────────────────────────────────────────

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
        logger.info("Page %d extracted successfully.", page_data["page_no"])
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

    # Filter out blank/None entries — FastAPI can pass [""] for an empty form field
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
