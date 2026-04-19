import os
import json
from dotenv import load_dotenv
from google import genai
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

try:
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR libraries not found. Install with: pip install pytesseract pdf2image Pillow")
    print("You also need Tesseract and Poppler installed on your system.\n")


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or ""
).strip().strip('"').strip("'")

if not api_key:
    raise ValueError("No API key found in .env (GOOGLE_API_KEY or GEMINI_API_KEY).")


PRIMARY_MODEL = "gemini-2.5-flash"
MEMORY_FILE = "chat_memory.json"
KNOWLEDGE_FOLDER = "knowledge"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

client = genai.Client(api_key=api_key)

history = []

if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r") as f:
            history = json.load(f)

        if not isinstance(history, list):
            history = []
    except:
        history = []


def ocr_image_file(filepath):
    if not OCR_AVAILABLE:
        print(f"  [OCR unavailable] Skipping image: {filepath}")
        return ""
    try:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"  [OCR error] {filepath}: {e}")
        return ""


def ocr_pdf_page(page_image):
    if not OCR_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(page_image)
        return text.strip()
    except Exception as e:
        print(f"  [OCR error on PDF page]: {e}")
        return ""


def extract_pdf_text(filepath, filename):
    result = ""
    try:
        reader = PdfReader(filepath)
        pdf_images = None
        if OCR_AVAILABLE:
            try:
                pdf_images = convert_from_path(filepath, dpi=300)
            except Exception as e:
                print(f"  [pdf2image error] Could not render pages for OCR fallback: {e}")

        for page_num, page in enumerate(reader.pages):
            page_label = f"\n--- Page {page_num + 1} ---\n"
            text = page.extract_text() or ""

            if text.strip():
                result += page_label + text
            elif pdf_images and page_num < len(pdf_images):
                # No text extracted — attempt OCR on the rendered page image
                print(f"  [OCR] Page {page_num + 1} of '{filename}' has no selectable text, running OCR...")
                ocr_text = ocr_pdf_page(pdf_images[page_num])
                if ocr_text:
                    result += page_label + f"[OCR]\n{ocr_text}"
                else:
                    result += page_label + "[No text found after OCR]\n"
            else:
                result += page_label + "[No text extractable from this page]\n"

    except Exception as e:
        print(f"Error reading PDF '{filename}': {e}")

    return result


def load_knowledge():
    knowledge_text = ""

    if not os.path.exists(KNOWLEDGE_FOLDER):
        print("No knowledge folder found.")
        return knowledge_text

    for filename in sorted(os.listdir(KNOWLEDGE_FOLDER)):
        filepath = os.path.join(KNOWLEDGE_FOLDER, filename)

        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()
        header = f"\n\n===== FILE: {filename} =====\n"

        try:
            if ext == ".txt":
                with open(filepath, "r", encoding="utf-8") as f:
                    knowledge_text += header + f.read()

            elif ext == ".pdf":
                print(f"Loading PDF: {filename}")
                knowledge_text += header + extract_pdf_text(filepath, filename)

            elif ext in IMAGE_EXTENSIONS:
                print(f"Loading image via OCR: {filename}")
                ocr_text = ocr_image_file(filepath)
                if ocr_text:
                    knowledge_text += header + f"[OCR]\n{ocr_text}"
                else:
                    knowledge_text += header + "[No text found after OCR]\n"

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return knowledge_text


knowledge_text = load_knowledge()

print("\n--- Loaded Knowledge Files ---")
for filename in sorted(os.listdir(KNOWLEDGE_FOLDER)) if os.path.exists(KNOWLEDGE_FOLDER) else []:
    if os.path.isfile(os.path.join(KNOWLEDGE_FOLDER, filename)):
        print(f"  - {filename}")
print("------------------------------\n")

chat = client.chats.create(
    model=PRIMARY_MODEL,
    history=[
        {"role": msg["role"], "parts": msg["parts"]}
        for msg in history
    ]
)

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f)


def ask_gemini(question):
    global history, chat

    prompt = f"""
You are an AI Teacher.

You MUST answer using the knowledge provided below if relevant.

KNOWLEDGE:
{knowledge_text}

QUESTION:
{question}
"""

    history.append({
        "role": "user",
        "parts": [{"text": question}]
    })

    response = chat.send_message(prompt)

    history.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })

    save_memory()

    return response.text


if __name__ == "__main__":
    print("Gemini AI Teacher (type 'exit' to quit)\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            save_memory()
            break

        try:
            answer = ask_gemini(question)

            print("\nGemini:")
            print(answer)
            print()

        except Exception as e:
            print("API error:", e)
