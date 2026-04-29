import pdfplumber
import pandas as pd
import cv2
import numpy as np
import requests
import base64
import json
import re
import os
import sys
import yaml
from PIL import Image
import io
import time


# ─────────────────────────────────────────────
#  CONFIG LOADER
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load and validate config.yaml.
    Required keys: api_url, fields, prompt (path to .txt file)

    input_folder and output_excel are NOT in the config —
    input_folder is asked at runtime, output_excel is auto-placed inside it.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = ["api_url", "fields", "prompt", "dpi_pdf"]
    missing = [k for k in required if k not in cfg or not cfg[k]]
    if missing:
        raise ValueError(f"config.yaml is missing required key(s): {missing}")

    if not isinstance(cfg["fields"], list) or len(cfg["fields"]) == 0:
        raise ValueError("config.yaml 'fields' must be a non-empty list.")

    # Load prompt from the .txt file path specified in config
    prompt_path = cfg["prompt"].strip()
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Support optional <<< >>> delimiters; otherwise use full file
    start = content.find("<<<")
    end   = content.find(">>>")
    if start != -1 and end != -1 and start < end:
        prompt_text = content[start + 3:end].strip()
    else:
        prompt_text = content.strip()

    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_path}")

    cfg["prompt_text"] = prompt_text
    return cfg


def ask_input_folder() -> str:
    """Ask the user for the input folder path at runtime."""
    supported = {".pdf", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    while True:
        folder = input("\nEnter folder path: ").strip().strip("\"'")
        if not folder:
            print("  Folder path cannot be empty.")
            continue
        if not os.path.exists(folder):
            print("  Folder not found.")
            continue
        if not os.path.isdir(folder):
            print("  That path is not a folder.")
            continue
        files = [
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f)[1].lower() in supported
        ]
        if not files:
            print(f"  No supported files found ({', '.join(sorted(supported))}).")
            continue
        print(f"  Found {len(files)} supported file(s).")
        return folder


def keys_to_column_headers(keys: list) -> list:
    """snake_case → Title Case  (e.g. 'invoice_no' → 'Invoice No')"""
    return [k.replace("_", " ").title() for k in keys]


def detect_color(image_rgb: np.ndarray, sensitivity: float = 0.5) -> bool:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    masks = [
        cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255])),
        cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255])),
        cv2.inRange(hsv, np.array([10, 50, 50]), np.array([25, 255, 255])),
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return (np.count_nonzero(combined) / combined.size * 100) > sensitivity


def preprocess_image(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        has_color = detect_color(image, sensitivity=0.3)
    else:
        gray = image
        has_color = False

    if has_color:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(binary, kernel, iterations=2)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(binary, kernel, iterations=1)


# ─────────────────────────────────────────────
#  QWEN OCR CLIENT
# ─────────────────────────────────────────────

def image_to_base64(image_input) -> str:
    if isinstance(image_input, str):
        with open(image_input, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    if isinstance(image_input, np.ndarray):
        _, buf = cv2.imencode(".jpg", image_input)
        return base64.b64encode(buf).decode("utf-8")
    if isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    raise TypeError(f"Unsupported image type: {type(image_input)}")


def call_qwen_api(
    api_url: str,
    image_b64,
    prompt: str,
    document_type: str = "General",
    fields: list = None,
    timeout: int = 60
) -> dict:
    """Send one or more base64 images to the Qwen OCR API.
    
    Payload format matches the server's MessageContent Pydantic model:
      {
        "document_type": "Invoice",
        "fields": ["invoice_number", "total"],  # optional
        "content": [
          {"type": "text", "text": "..."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ]
      }
    """
    images = [image_b64] if isinstance(image_b64, str) else list(image_b64)

    content = [{"type": "text", "text": prompt}]
    for b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    payload = {
        "document_type": document_type,
        "content": content
    }
    if fields:
        payload["fields"] = fields

    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()

        if result.get("success"):
            raw = result.get("data", {})
            print(raw)
            return _parse_json_from_text(raw) if isinstance(raw, str) else raw
        else:
            raise RuntimeError(f"API returned success=False: {result}")

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Qwen API at {api_url}.\n"
            "Make sure the server is running."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Qwen API timed out after {timeout}s.")


def _parse_json_from_text(text: str) -> dict:
    """Extract a JSON object from raw text (handles markdown fences etc.)."""
    text = re.sub(r'^```[a-zA-Z]*\n?', '', text.strip())
    text = re.sub(r'\n?```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}



# ─────────────────────────────────────────────
#  CORE EXTRACTOR
# ─────────────────────────────────────────────

class DocumentExtractor:
    """
    Generic document extractor powered by Qwen Vision API.
    All configuration comes from config.yaml — no interactive input needed.

    Supports: PDF, Excel, PNG, JPG, TIFF, BMP, WEBP.
    Output Excel columns are driven by the 'fields' list in config.yaml.
    """

    def __init__(self, cfg: dict):
        self.api_url       = cfg["api_url"]
        self.prompt        = cfg["prompt"]
        self.fields        = cfg["fields"]
        self.dpi_pdf       = cfg["dpi_pdf"]
        self.document_type = cfg.get("document_type", "General")
        self.col_headers   = keys_to_column_headers(self.fields)

        print(f"  Fields        : {self.fields}")
        print(f"  Document Type : {self.document_type}")
        print(f"  Columns       : {self.col_headers}")

    # ── helpers ──────────────────────────────

    def _empty_record(self) -> dict:
        return {f: "" for f in self.fields}

    def _normalize(self, raw: dict) -> dict:
        """Keep only fields defined in config, fill missing ones with empty string."""
        record = self._empty_record()
        for k, v in raw.items():
            if k in record:
                record[k] = v if v is not None else ""
        return record

    # ── image files ──────────────────────────

    def extract_from_image_file(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        b64     = image_to_base64(img_rgb)
        raw     = call_qwen_api(
            self.api_url, b64, self.prompt,
            document_type=self.document_type,
            fields=self.fields
        )
        return self._normalize(raw)

    # ── PDF files ────────────────────────────
    def extract_from_pdf(self, pdf_path: str) -> dict:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            head = list(range(min(2, total_pages)))
            tail = list(range(max(total_pages - 2, 2), total_pages))
            page_indices = sorted(set(head + tail))

            print(f"  → PDF has {total_pages} page(s). Scanning: "
                  f"{[i + 1 for i in page_indices]}")

            page_images_b64 = []
            combined_text = ""

            for page_num in page_indices:
                page = pdf.pages[page_num]
                text = page.extract_text() or ""
                print(len(text.strip()))
                if len(text.strip()) > 1200:
                    print(f"  → Page {page_num + 1}: text-rich")
                    combined_text += f"\n--- Page {page_num + 1} ---\n{text}"
                else:
                    print(f"  → Page {page_num + 1}: image-based, rasterising…")
                    try:
                        pil_img = page.to_image(resolution=self.dpi_pdf).original
                        prepro_img = preprocess_image(pil_img)
                        page_images_b64.append(image_to_base64(prepro_img))
                    except Exception as e:
                        print(f"Rasterise error page {page_num + 1}: {e}")

            # Single API call with all images
            if page_images_b64:
                prompt = self.prompt
                if combined_text:
                    prompt += f"\n\nADDITIONAL TEXT FROM DOCUMENT:\n{combined_text}"
                print(f"  → Sending {len(page_images_b64)} image(s) to API…")
                try:
                    raw = call_qwen_api(
                        self.api_url,
                        page_images_b64,
                        prompt,
                        document_type=self.document_type,
                        fields=self.fields
                    )
                    return self._normalize(raw)
                except requests.exceptions.HTTPError as e:
                    detail = e.response.json().get("detail", {})
                    print(f"Http error [{type(e).__name__}]: {detail['message']}")
                    return self._empty_record()
                except Exception as e:
                    print(f"Unexpected error [{type(e).__name__}]: {e}")
                    return self._empty_record()

            elif combined_text:
                text_prompt = self.prompt + f"\n\nDOCUMENT TEXT:\n{combined_text}"
                payload = {
                    "document_type": self.document_type,
                    "content": [{"type": "text", "text": text_prompt}]
                }
                if self.fields:
                    payload["fields"] = self.fields
                try:
                    resp = requests.post(self.api_url, json=payload, timeout=60)
                    resp.raise_for_status()
                    r = resp.json()
                    raw = r.get("data", {}) if r.get("success") else {}
                    if isinstance(raw, str):
                        raw = _parse_json_from_text(raw)
                    return self._normalize(raw)
                except Exception as e:
                    print(f" Text API error: {e}")
                    return self._empty_record()

        return self._empty_record()

    # ── Excel files ──────────────────────────

    def extract_from_excel(self, excel_path: str) -> dict:
        all_text = ""
        xf = pd.ExcelFile(excel_path)
        for sheet in xf.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            all_text += f"\n=== Sheet: {sheet} ===\n{df.to_string(index=False)}\n"

        text_prompt = self.prompt + f"\n\nSPREADSHEET TEXT:\n{all_text[:8000]}"
        payload = {
            "document_type": self.document_type,
            "content": [{"type": "text", "text": text_prompt}]
        }
        if self.fields:
            payload["fields"] = self.fields
        resp = requests.post(self.api_url, json=payload, timeout=60)
        resp.raise_for_status()
        r   = resp.json()
        raw = r.get("data", {}) if r.get("success") else {}
        if isinstance(raw, str):
            raw = _parse_json_from_text(raw)
        return self._normalize(raw)

    # ── dispatcher ───────────────────────────

    def process_file(self, file_path: str) -> dict:
        t0  = time.perf_counter()
        ext = os.path.splitext(file_path)[1].lower()
        print(f"  → Processing ({ext}): {os.path.basename(file_path)}")

        if ext == ".pdf":
            data = self.extract_from_pdf(file_path)
        elif ext in (".xlsx", ".xls"):
            data = self.extract_from_excel(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            data = self.extract_from_image_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        print(f"[TIME] {time.perf_counter() - t0:.3f}s")
        return data

    # ── folder processor ─────────────────────

    def process_folder(self, input_folder: str, output_excel: str) -> pd.DataFrame:
        supported = {".pdf", ".xlsx", ".xls", ".png", ".jpg",
                     ".jpeg", ".tiff", ".bmp", ".webp"}

        files = [
            f for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f))
               and os.path.splitext(f)[1].lower() in supported
               and f != "extracted_results.xlsx"
        ]

        if not files:
            print(f"No supported files found in: {input_folder}")
            return pd.DataFrame()

        print(f"\nFound {len(files)} file(s) to process")
        print("=" * 80)

        column_order = ["File Name"] + self.col_headers + ["Status"]
        results      = []

        for idx, filename in enumerate(files, 1):
            t_file    = time.perf_counter()
            file_path = os.path.join(input_folder, filename)
            print(f"\n[{idx}/{len(files)}] {filename}")
            print("-" * 60)

            try:
                data = self.process_file(file_path)
                row  = {"File Name": filename, "Status": "OK"}

                for field, col in zip(self.fields, self.col_headers):
                    value    = data.get(field, "")
                    row[col] = value if value != "" else None
                    print(f"  ✓ {col:<30}: {str(value)[:60]}")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                row = {"File Name": filename, "Status": f"ERROR: {e}"}
                for col in self.col_headers:
                    row[col] = None

            results.append(row)
            print(f"[TIME] File total: {time.perf_counter() - t_file:.3f}s")

        df = pd.DataFrame(results)
        for col in column_order:
            if col not in df.columns:
                df[col] = None
        df = df[column_order]

        os.makedirs(os.path.dirname(os.path.abspath(output_excel)), exist_ok=True)
        df.to_excel(output_excel, index=False, engine="openpyxl")

        print(f"\n{'=' * 80}")
        print(f"✓ Output saved → {output_excel}")
        return df


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    print("=" * 60)
    print("  DOCUMENT EXTRACTION TOOL  (Qwen Vision API)")
    print(f"  Config : {config_path}")
    print("=" * 60)

    try:
        cfg = load_config(config_path)

        cfg["prompt"] = cfg["prompt_text"]

        print(f"\n  API URL : {cfg['api_url']}")
        print(f"  Fields  : {cfg['fields']}")
        print(f"  PDF DPI  : {cfg['dpi_pdf']}")

        input_folder = ask_input_folder()
        output_excel = os.path.join(input_folder, "extracted_results.xlsx")

        print(f"\n  Input folder : {input_folder}")
        print(f"  Output Excel : {output_excel}")

        extractor = DocumentExtractor(cfg)

        print("\n" + "=" * 60)
        print("  PROCESSING FILES…")
        print("=" * 60)

        t_start = time.perf_counter()
        df = extractor.process_folder(input_folder, output_excel)

        print(f"\n{'=' * 60}")
        print("  DONE!")
        print(f"  Files processed : {len(df)}")
        print(f"  Output          : {output_excel}")
        print(f"  Total time      : {time.perf_counter() - t_start:.1f}s")
        print("=" * 60)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ✗ Config error: {e}")
    except KeyboardInterrupt:
        print("\n\n  Cancelled by user.")
    except Exception as e:
        print(f"\n  ✗ Fatal error: {e}")
    finally:
        input("\nPress Enter to exit…")