# 📄 Document OCR API Documentation

## 📘 Overview
The **Document OCR API** is a FastAPI-based service that extracts structured JSON data from document images using a Vision Language Model (`qwen3-vl`). It supports specialized extraction for Indonesian documents (KTP, KK, NPWP, SIM) and general business documents (Invoices, Quotations). All images must be sent as base64-encoded data URIs, and the API handles image preprocessing, prompt routing, and strict JSON formatting.

---

## 🌐 Base URL
```
http://192.168.13.176:5030
```
*(Adjust host/port based on your deployment configuration)*

---

## 📡 Endpoints

### 1. `GET /`
**Description:** Returns basic API metadata, version, and available route information.
- **Method:** `GET`
- **Authentication:** None
- **Response:**
```json
{
  "name": "Document OCR API",
  "version": "2.0.0",
  "endpoints": {
    "image_ocr_process": "/ocr/process",
    "health": "/health"
  },
  "docs": "/docs"
}
```

**Example Usage:**
```bash
curl http://192.168.13.176:5030/
```

---

### 2. `GET /health`
**Description:** Checks the connectivity and readiness of the backend vLLM model server (`http://192.168.13.176:8053`). Useful for monitoring and deployment health probes.
- **Method:** `GET`
- **Authentication:** None
- **Response:**
```json
{
  "status": "Healthy",
  "model_ready": true,
  "vllm_max_model_len": 11000,
  "width": 512,
  "height": 512
}
```
*Note: Returns `Unhealthy`, `vLLM server not reachable`, or `vLLM server timeout` with `model_ready: false` on failure.*

**Example Usage:**
```bash
curl http://192.168.13.176:5030/health
```

---

### 3. `POST /ocr/process`
**Description:** Core endpoint. Accepts one or more document images, applies a specialized system prompt based on `document_type`, and returns extracted data as structured JSON.
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Authentication:** None (API key is handled internally for vLLM)

#### 🔹 Request Body Schema
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_type` | `string` (enum) | No | Selects the extraction prompt. Options: `General`, `KTP`, `KK`, `NPWP`, `Invoice`, `Quotation`, `SIM`. Defaults to `General`. |
| `fields` | `array[string]` | No | List of exact `snake_case` keys to extract. **Overrides** default fields if provided. |
| `content` | `array[object]` | Yes | Ordered list of text instructions and/or images. **Max 5 images**. External URLs are rejected. |

**`content` Item Format:**
- Text: `{"type": "text", "text": "Your custom instruction"}`
- Image: `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`

#### 🔹 Success Response
```json
{
  "success": true,
  "data": {
    "invoice_number": "INV-2024-001",
    "total": "Rp 1.500.000",
    "line_items": [ ... ]
  }
}
```

#### 🔹 Error Response Format
```json
{
  "success": false,
  "error_type": "json_parse_error",
  "message": "Model returned malformed JSON",
  "detail": "..."
}
```

---

## 📦 Supported Document Types & Default Fields
The API automatically maps `document_type` to optimized system prompts. Default extraction fields include:

| Type | Key Fields Extracted |
|------|----------------------|
| `General` | Flat structure or arrays for tables, auto-detected hierarchy |
| `KTP` | `province`, `city`, `nik`, `full_name`, `birth_place`, `gender`, `address`, `religion`, `marital_status`, `occupation`, etc. |
| `KK` | Top-level: `kk_number`, `head_of_family`, address fields. `members[]` array with personal/education/occupation data |
| `NPWP` | `npwp_number`, `full_name`, `address`, `registration_date`, `kpp_office` |
| `Invoice` | Header info + `line_items[]` array (`description`, `quantity`, `unit_price`, `amount`, etc.) |
| `Quotation` | Header info + `line_items[]` array, `terms_and_conditions`, `valid_until` |
| `SIM` | `name`, `alamat`, `tempat_tanggal_lahir`, `nomor_sim`, `jenis_sim`, `berlaku_sampai`, `pekerjaan` |

---

## 🚀 Usage Examples

### 🔹 cURL Example
```bash
curl -X POST http://localhost:5030/ocr/process \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "Invoice",
    "fields": ["invoice_number", "total", "buyer_name"],
    "content": [
      {"type": "text", "text": "Extract only the requested fields."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."}}
    ]
  }'
```

### 🔹 Python (`requests`) Example
```python
import requests
import base64

# 1. Convert local image to base64 data URI
with open("invoice.jpg", "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")
    image_uri = f"data:image/jpeg;base64,{b64_data}"

payload = {
    "document_type": "Invoice",
    "fields": ["invoice_number", "total_amount", "invoice_date"],
    "content": [
        {"type": "text", "text": "Extract the specified fields as strings."},
        {"type": "image_url", "image_url": {"url": image_uri}}
    ]
}

response = requests.post("http://localhost:5030/ocr/process", json=payload)
print(response.json())
```

---

## ⚠️ Error Handling & Status Codes
| HTTP Status | `error_type` | Description |
|-------------|--------------|-------------|
| `400` | `url_not_allowed` | External `http/https` URLs detected. Use base64 data URIs. |
| `400` | `image_limit_exceeded` | More than 5 images provided in one request. |
| `400` | `llm_bad_request` | Input too long for vLLM context. Resize images or reduce text. |
| `401` | `llm_auth_error` | vLLM API key misconfigured. |
| `422` | `json_parse_error` | LLM returned non-JSON or malformed JSON. |
| `429` | `llm_rate_limit` | vLLM server overloaded. Retry after a short delay. |
| `502` | `llm_server_error` | vLLM returned 500/503. Check backend logs. |
| `503` | `llm_unreachable` | Cannot connect to vLLM server. |
| `504` | `llm_timeout` | LLM processing took too long. Use smaller images. |
| `500` | `internal_error` | Unexpected FastAPI/server error. |

---

## 📌 Important Notes & Limitations
1. **Base64 Only:** External URLs (`http://` or `https://`) are strictly rejected for security and latency reasons.
2. **Max Images:** Maximum **5 images** per request. Exceeding this returns a `400` error.
3. **Image Preprocessing:** Images are automatically resized to a maximum of `1024x1024` px, converted to RGB, and saved as JPEG (95% quality) before being sent to the model.
4. **Field Override:** If `fields` is provided, the AI **only** extracts those exact keys. Missing fields are returned as `""`.
5. **Verbatim Extraction:** Text is extracted exactly as written. No normalization, spelling correction, or hallucination is allowed.
6. **Interactive Docs:** Visit `/docs` (Swagger UI) or `/redoc` for an interactive testing interface.

---