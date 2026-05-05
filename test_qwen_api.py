"""
Comprehensive test suite for Qwen OCR API.
Tests all possible input scenarios: success cases, validation errors, and edge cases.

Usage:
    1. Start your API server:  python qwen_api.py
    2. Run tests:              python test_qwen_api.py
"""
import requests
import base64
import os
import json

API_URL = "http://localhost:5030/ocr/process"
API_UPLOAD_URL = "http://localhost:5030/ocr/process/upload"
API_HEALTH = "http://localhost:5030/health"
API_ROOT = "http://localhost:5030/"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_dummy_image_bytes(format="JPEG"):
    from PIL import Image as PILImage
    import io as _io
    buf = _io.BytesIO()
    PILImage.new("RGB", (10, 10), "white").save(buf, format=format)
    return buf.getvalue()

def encode_image(path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def print_result(response):
    """Pretty-print the API response."""
    print(f"  Status Code : {response.status_code}")
    try:
        body = response.json()
        print(f"  Response    : {json.dumps(body, indent=2, ensure_ascii=False)[:500]}")
    except Exception:
        print(f"  Raw Body    : {response.text[:500]}")


def run_test(name: str, func):
    """Run a single test with header and separator."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        func()
    except Exception as e:
        print(f"  TEST RUNNER ERROR: {e}")


# ---------------------------------------------------------------------------
# SUCCESS CASES (require vLLM to be running for 200; otherwise expect 502/503)
# ---------------------------------------------------------------------------

def test_01_general_base64():
    """Test 1: General OCR with IJAZAH.jpg via JSON base64"""
    img_path = "IJAZAH.jpg"
    if not os.path.exists(img_path):
        print(f"  SKIP: {img_path} not found"); return

    payload = {
        "document_type": "General",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)


def test_02_ktp_base64_no_prompt():
    """Test 2: KTP OCR with ktp_1.png, NO text prompt (auto-inject default)"""
    img_path = os.path.join("KTP", "ktp_1.png")
    if not os.path.exists(img_path):
        print(f"  SKIP: {img_path} not found"); return

    payload = {
        "document_type": "KTP",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(img_path)}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)


def test_03_invoice_base64_with_prompt():
    """Test 3: Invoice OCR with invoice_example.PNG + custom prompt"""
    img_path = "invoice_example.PNG"
    if not os.path.exists(img_path):
        print(f"  SKIP: {img_path} not found"); return

    payload = {
        "document_type": "Invoice",
        "content": [
            {"type": "text", "text": "Extract all line items and totals from this invoice"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(img_path)}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)


def test_04_npwp_base64():
    """Test 4: NPWP OCR with NPWP.png"""
    img_path = os.path.join("npwp", "NPWP.png")
    if not os.path.exists(img_path):
        print(f"  SKIP: {img_path} not found"); return

    payload = {
        "document_type": "NPWP",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(img_path)}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)

def test_05_kk_base64():
    """Test 5: KK (Kartu Keluarga) OCR"""
    img_path = os.path.join("KK", "kk_1.png")
    if not os.path.exists(img_path):
        b64 = base64.b64encode(create_dummy_image_bytes()).decode("utf-8")
        print(f"  NOTE: {img_path} not found, using a dummy image to test routing.")
    else:
        b64 = encode_image(img_path)

    payload = {
        "document_type": "KK",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)

def test_06_sim_base64():
    """Test 6: SIM OCR"""
    img_path = os.path.join("SIM", "sim_1.png")
    if not os.path.exists(img_path):
        from PIL import Image as PILImage
        import io as _io
        buf = _io.BytesIO()
        PILImage.new("RGB", (10, 10), "white").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        print(f"  NOTE: {img_path} not found, using a dummy image to test routing.")
    else:
        b64 = encode_image(img_path)

    payload = {
        "document_type": "SIM",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)

def test_07_quotation_base64():
    """Test 7: Quotation OCR"""
    img_path = "quotation_example.PNG"
    if not os.path.exists(img_path):
        from PIL import Image as PILImage
        import io as _io
        buf = _io.BytesIO()
        PILImage.new("RGB", (10, 10), "white").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        print(f"  NOTE: {img_path} not found, using a dummy image to test routing.")
    else:
        b64 = encode_image(img_path)

    payload = {
        "document_type": "Quotation",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)

def test_07b_stnk_base64():
    """Test 7b: STNK OCR"""
    b64 = base64.b64encode(create_dummy_image_bytes()).decode("utf-8")
    payload = {
        "document_type": "STNK",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)

def test_07c_passport_base64():
    """Test 7c: Passport OCR"""
    b64 = base64.b64encode(create_dummy_image_bytes()).decode("utf-8")
    payload = {
        "document_type": "Passport",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)

def test_08_fields_extraction():
    """Test 8: Custom Fields Extraction"""
    img_path = os.path.join("KTP", "ktp_1.png")
    if not os.path.exists(img_path):
        from PIL import Image as PILImage
        import io as _io
        buf = _io.BytesIO()
        PILImage.new("RGB", (10, 10), "white").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        print(f"  NOTE: {img_path} not found, using a dummy image to test routing.")
    else:
        b64 = encode_image(img_path)

    payload = {
        "document_type": "KTP",
        "fields": ["nik", "nama", "tanggal_lahir"],
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)


# ---------------------------------------------------------------------------
# ERROR / VALIDATION CASES (these should NOT reach vLLM)
# ---------------------------------------------------------------------------

def test_09_url_rejected():
    """Test 9: External URL should be REJECTED with 400"""
    payload = {
        "document_type": "General",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print("  PASS: URL correctly rejected")


def test_10_image_limit_exceeded():
    """Test 10: Sending 6 images should be REJECTED with 400 (max is 5)"""
    # Create a tiny valid base64 image
    tiny_b64 = base64.b64encode(create_dummy_image_bytes()).decode("utf-8")

    payload = {
        "document_type": "General",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tiny_b64}"}}
            for _ in range(6)  # 6 images = over the limit
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    assert "image_limit_exceeded" in r.text
    print("  PASS: 6-image limit correctly enforced")


def test_11_empty_content():
    """Test 11: Empty content list should fail Pydantic validation (422)"""
    payload = {
        "document_type": "General",
        "content": []
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)
    # FastAPI/Pydantic may return 422 for validation or the server may handle differently
    assert r.status_code in (400, 422), f"Expected 400 or 422, got {r.status_code}"
    print("  PASS: Empty content correctly rejected")


def test_12_invalid_document_type():
    """Test 12: Invalid document_type 'UnknownType' should fail Pydantic validation (422)"""
    tiny_b64 = base64.b64encode(create_dummy_image_bytes()).decode("utf-8")

    payload = {
        "document_type": "UnknownType",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tiny_b64}"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("  PASS: Invalid document_type correctly rejected")


def test_13_invalid_base64():
    """Test 13: Corrupted base64 string should fail during preprocessing"""
    payload = {
        "document_type": "General",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,NOT_VALID_BASE64_!!!"}}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)
    assert r.status_code in (400, 500), f"Expected 400 or 500, got {r.status_code}"
    print("  PASS: Invalid base64 correctly handled")


def test_14_no_image_provided():
    """Test 14: Text only (no image) should be REJECTED with 400"""
    payload = {
        "document_type": "General",
        "content": [
            {"type": "text", "text": "Extract fields"}
        ]
    }
    r = requests.post(API_URL, json=payload)
    print_result(r)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print("  PASS: No image provided correctly handled")


# ---------------------------------------------------------------------------
# MULTIPART UPLOAD CASES
# ---------------------------------------------------------------------------

def test_15_upload_success():
    """Test 15: Multipart upload success"""
    files = [
        ("files", ("NPWP.png", open(r"npwp\NPWP.png", "rb").read(), "image/png"))
    ]
    data = {"document_type": "NPWP"} 
    r = requests.post(API_UPLOAD_URL, files=files, data=data)
    print_result(r)

def test_16_upload_with_fields():
    """Test 16: Multipart upload with custom fields and prompt"""
    files = [
        ("files", ("ktp_1.png", open(r"KTP\ktp_1.png", "rb").read(), "image/png"))
    ]
    data = {
        "document_type": "KTP",
        "fields": ["nik", "nama"],
        "custom_prompt": "Extract the NIK and Name clearly."
    }
    r = requests.post(API_UPLOAD_URL, files=files, data=data)
    print_result(r)

def test_16b_upload_multiple_images():
    """Test 16b: Multipart upload with multiple images"""
    files = [
        ("files", ("NPWP.png", open(r"npwp\NPWP.png", "rb").read(), "image/png")),
        ("files", ("ktp_1.png", open(r"KTP\ktp_1.png", "rb").read(), "image/png"))
    ]
    data = {
        "document_type": "General",
        "custom_prompt": "Extract all information from these documents."
    }
    r = requests.post(API_UPLOAD_URL, files=files, data=data)
    print_result(r)

def test_16c_upload_pdf():
    """Test 16c: Multipart upload with PDF file (converted to image first)"""
    import pdfplumber
    import io
    pdf_path = r"invoice_contoh\20260108-2026GI000000114.pdf"
    if not os.path.exists(pdf_path):
        print(f"  SKIP: {pdf_path} not found")
        return

    # Convert all pages of the PDF to images
    files = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                pil_img = page.to_image(resolution=300).original
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                files.append(("files", (f"invoice_page{i+1}.png", img_bytes, "image/png")))
    except Exception as e:
        print(f"  ERROR processing PDF: {e}")
        return

    if not files:
        print("  SKIP: No pages found in PDF")
        return
    data = {
        "document_type": "Invoice",
        "fields": ["invoice_number", "sales_order", "date", "total_amount"]
    }
    r = requests.post(API_UPLOAD_URL, files=files, data=data)
    print_result(r)

def test_17_upload_unsupported_type():
    """Test 17: Multipart upload with unsupported file type (415)"""
    files = [
        ("files", ("dummy.txt", b"Hello World", "text/plain"))
    ]
    data = {"document_type": "General"}
    r = requests.post(API_UPLOAD_URL, files=files, data=data)
    print_result(r)
    assert r.status_code == 415, f"Expected 415, got {r.status_code}"
    print("  PASS: Unsupported type correctly rejected")

def test_18_root_endpoint():
    """Test 18: Root endpoint returns API info"""
    r = requests.get(API_ROOT)
    print_result(r)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print("  PASS: Root endpoint works")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  QWEN OCR API — COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    # Check server health first
    try:
        h = requests.get(API_HEALTH, timeout=3)
        print(f"\n  Server health: {h.json().get('status', 'unknown')}")
    except Exception:
        print("\n  WARNING: API server may not be running on localhost:5030!")

    # # --- Success cases (need vLLM for full 200, but test API layer regardless) ---
    # run_test("Test 01: General OCR — IJAZAH.jpg (base64 JSON)", test_01_general_base64)
    # run_test("Test 02: KTP OCR — ktp_1.png (base64, no prompt)", test_02_ktp_base64_no_prompt)
    # run_test("Test 03: Invoice OCR — invoice_example.PNG (base64 + prompt)", test_03_invoice_base64_with_prompt)
    # run_test("Test 04: NPWP OCR — NPWP.png (base64)", test_04_npwp_base64)
    # run_test("Test 05: KK OCR (base64)", test_05_kk_base64)
    # run_test("Test 06: SIM OCR (base64)", test_06_sim_base64)
    # run_test("Test 07: Quotation OCR (base64)", test_07_quotation_base64)
    # run_test("Test 08: Fields Extraction (KTP)", test_08_fields_extraction)

    # # --- Error/validation cases (should NOT reach vLLM) ---
    # run_test("Test 09: REJECT external URL", test_09_url_rejected)
    # run_test("Test 10: REJECT >5 images", test_10_image_limit_exceeded)
    # run_test("Test 11: REJECT empty content", test_11_empty_content)
    # run_test("Test 12: REJECT invalid document_type", test_12_invalid_document_type)
    # run_test("Test 13: REJECT corrupted base64", test_13_invalid_base64)
    # run_test("Test 14: REJECT no image provided", test_14_no_image_provided)

    # --- Multipart upload cases ---
    # run_test("Test 15: Upload success", test_15_upload_success)
    # run_test("Test 16: Upload with fields and custom prompt", test_16_upload_with_fields)
    # run_test("Test 16b: Upload multiple images", test_16b_upload_multiple_images)
    # run_test("Test 16c: Upload PDF file", test_16c_upload_pdf)
    # run_test("Test 17: Upload unsupported type", test_17_upload_unsupported_type)

    # --- Other endpoints ---
    run_test("Test 18: Root endpoint", test_18_root_endpoint)

    print(f"\n{'='*60}")
    print("  ALL TESTS DISPATCHED")
    print(f"{'='*60}")
