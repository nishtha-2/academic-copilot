from __future__ import annotations

import io
import os
import re
import shutil
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def configure_tesseract() -> None:
    """
    Resolve Tesseract without hardcoding one mandatory installation path.

    Priority:
    1. TESSERACT_CMD environment variable
    2. executable available in PATH
    3. common macOS and Windows installation paths
    """
    configured = os.getenv("TESSERACT_CMD", "").strip()

    if configured and Path(configured).exists():
        pytesseract.pytesseract.tesseract_cmd = configured
        return

    system_command = shutil.which("tesseract")

    if system_command:
        pytesseract.pytesseract.tesseract_cmd = system_command
        return

    candidates = (
        # Apple Silicon Mac
        Path("/opt/homebrew/bin/tesseract"),

        # Intel Mac
        Path("/usr/local/bin/tesseract"),

        # Windows
        Path(
            r"C:\Users\Prachi.Bhardwaj\AppData\Local\Programs"
            r"\Tesseract-OCR\tesseract.exe"
        ),
    )

    for candidate in candidates:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return

    raise RuntimeError(
        "Tesseract was not found. Install it using Homebrew, add it to PATH, "
        "or set TESSERACT_CMD to the full Tesseract executable path."
    )
def _clean_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Generic preprocessing for diagrams, screenshots and textbook figures.
    """
    processed = image.convert("L")

    if processed.width < 1500:
        scale = 1500 / max(processed.width, 1)
        processed = processed.resize(
            (
                int(processed.width * scale),
                int(processed.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )

    processed = ImageOps.autocontrast(processed)
    processed = ImageEnhance.Contrast(processed).enhance(1.9)
    processed = processed.filter(ImageFilter.SHARPEN)

    return processed


def _read_ocr(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        processed = _preprocess_for_ocr(image)

        text = pytesseract.image_to_string(
            processed,
            lang="eng",
            config="--oem 3 --psm 11",
        )

        return _clean_text(text)
    except Exception:
        return ""


def _get_image_rect(
    page: fitz.Page,
    xref: int,
) -> fitz.Rect | None:
    try:
        rectangles = page.get_image_rects(xref)

        if rectangles:
            return rectangles[0]
    except Exception:
        pass

    return None


def _extract_region_text(
    page: fitz.Page,
    rectangle: fitz.Rect,
) -> str:
    try:
        return _clean_text(
            page.get_text(
                "text",
                clip=rectangle,
            )
        )
    except Exception:
        return ""


def _extract_caption_and_title(
    page: fitz.Page,
    image_rect: fitz.Rect | None,
) -> tuple[str, str, str]:
    """
    Read text immediately above and below an image.

    Returns:
        title:
            Short likely figure title.
        caption:
            Full nearby caption text.
        nearby_text:
            Larger surrounding page text used for semantic ranking.
    """
    if image_rect is None:
        return "", "", ""

    page_rect = page.rect

    horizontal_padding = max(25, image_rect.width * 0.08)

    below = fitz.Rect(
        max(page_rect.x0, image_rect.x0 - horizontal_padding),
        image_rect.y1,
        min(page_rect.x1, image_rect.x1 + horizontal_padding),
        min(
            page_rect.y1,
            image_rect.y1 + max(140, image_rect.height * 0.45),
        ),
    )

    above = fitz.Rect(
        max(page_rect.x0, image_rect.x0 - horizontal_padding),
        max(
            page_rect.y0,
            image_rect.y0 - max(110, image_rect.height * 0.28),
        ),
        min(page_rect.x1, image_rect.x1 + horizontal_padding),
        image_rect.y0,
    )

    surrounding = fitz.Rect(
        max(page_rect.x0, image_rect.x0 - horizontal_padding),
        max(page_rect.y0, image_rect.y0 - 160),
        min(page_rect.x1, image_rect.x1 + horizontal_padding),
        min(page_rect.y1, image_rect.y1 + 220),
    )

    below_text = _extract_region_text(page, below)
    above_text = _extract_region_text(page, above)
    nearby_text = _extract_region_text(page, surrounding)

    figure_pattern = re.compile(
        r"\b(?:fig(?:ure)?|diagram|schematic|architecture|structure|"
        r"block\s+diagram|flowchart|circuit)\b",
        re.IGNORECASE,
    )

    caption_candidates = [
        text
        for text in (below_text, above_text)
        if text
    ]

    caption = ""

    for candidate in caption_candidates:
        if figure_pattern.search(candidate):
            caption = candidate
            break

    if not caption and caption_candidates:
        caption = caption_candidates[0]

    title = ""

    if caption:
        # Prefer the first short sentence/line-like segment as title.
        segments = re.split(r"(?<=[.!?])\s+|\s{2,}", caption)

        for segment in segments:
            cleaned = segment.strip(" -:;")

            if 3 <= len(cleaned) <= 180:
                title = cleaned
                break

    return title, caption, nearby_text


def extract_images(
    pdf_path: str | Path,
    output_folder: str | Path,
) -> list[dict[str, Any]]:
    """
    Extract meaningful PDF images and enrich each one with:
    - probable title
    - nearby caption
    - Tesseract OCR labels
    - surrounding page text
    - source/page/dimensions

    No subject-specific keywords are used.
    """
    configure_tesseract()

    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    document = fitz.open(str(pdf_path))
    results: list[dict[str, Any]] = []

    for page_index in range(len(document)):
        page = document[page_index]
        page_number = page_index + 1

        for image_index, image_info in enumerate(
            page.get_images(full=True)
        ):
            xref = image_info[0]

            try:
                extracted = document.extract_image(xref)
            except Exception:
                continue

            image_bytes = extracted.get("image")
            extension = extracted.get("ext", "png")
            width = int(extracted.get("width", 0) or 0)
            height = int(extracted.get("height", 0) or 0)
            area = width * height

            if not image_bytes:
                continue

            # Reject likely bullets, icons and tiny decorations.
            if width < 180 or height < 120 or area < 35_000:
                continue

            image_rect = _get_image_rect(page, xref)

            title, caption, nearby_text = (
                _extract_caption_and_title(
                    page,
                    image_rect,
                )
            )

            ocr_text = _read_ocr(image_bytes)

            searchable_text = _clean_text(
                " ".join(
                    value
                    for value in (
                        title,
                        caption,
                        ocr_text,
                        nearby_text,
                    )
                    if value
                )
            )

            filename = (
                f"{pdf_path.stem}"
                f"_page_{page_number}"
                f"_image_{image_index}.{extension}"
            )

            image_path = output_folder / filename
            image_path.write_bytes(image_bytes)

            results.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "area": area,
                    "title": title,
                    "caption": caption,
                    "ocr_text": ocr_text,
                    "nearby_text": nearby_text,
                    "searchable_text": searchable_text,
                }
            )

    document.close()
    return results
