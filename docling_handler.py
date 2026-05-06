import os
import tempfile
from typing import Any, Dict, List, TypedDict
from config import logger
from utils import _pil_to_content_item

# ── Docling availability check ─────────────────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, EasyOcrOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.datamodel.base_models import InputFormat
    _DOCLING_AVAILABLE = True
    logger.info("Docling is available — PDF text extraction enabled.")
except ImportError:
    _DOCLING_AVAILABLE = False
    logger.warning("Docling not installed. PDFs will fall back to image-based OCR.")


class PageData(TypedDict):
    page_no: int
    image: Dict[str, Any]               # Full page base64 image_url content dict
    table_images: List[Dict[str, Any]]   # Table crop base64 image_url content dicts
    markdown: str                        # Extracted text for this page


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
