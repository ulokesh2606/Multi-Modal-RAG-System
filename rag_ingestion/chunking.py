import re
import logging
from config import MIN_CHUNK_LENGTH, MIN_ALPHA_RATIO, TXT_CHUNK_SIZE, TXT_CHUNK_OVERLAP

logger = logging.getLogger("rag.chunking")


def chunk_by_sliding_window(text: str,
                             chunk_size: int = TXT_CHUNK_SIZE,
                             overlap: int = TXT_CHUNK_OVERLAP) -> list[dict]:
    """
    Sliding-window chunker for flat text files (txt, md, csv).

    Why this exists:
        chunk_by_title() relies on Unstructured detecting Title elements.
        Plain .txt files have none — everything is NarrativeText — so the
        title-based splitter either produces one giant blob or splits at
        arbitrary element boundaries that destroy sentence context.
        The result is reranker scores around -11 (vs normal 0–10) because
        the embedded text is incoherent.

    Strategy:
        1. Split on paragraph boundaries (double newline) first — preserves
           natural topic breaks without any model calls.
        2. If a paragraph is still larger than chunk_size, split further on
           sentence boundaries (period/question/exclamation + whitespace).
        3. Greedily pack sentences into chunks up to chunk_size characters.
        4. Roll the last `overlap` characters of each chunk into the next one
           so facts that straddle a boundary are still retrievable.

    Returns a list of dicts compatible with extract_original() output, so the
    rest of the pipeline (summarizer, vectorstore, validate_chunk) is unchanged.
    """
    text = clean_text(text)
    if not text:
        return []

    # ── Step 1: split into paragraphs ─────────────────────────────────────────
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # ── Step 2: split oversized paragraphs into sentences ─────────────────────
    sentences: list[str] = []
    _sent_re = re.compile(r"(?<=[.!?])\s+")
    for para in paragraphs:
        if len(para) <= chunk_size:
            sentences.append(para)
        else:
            parts = _sent_re.split(para)
            sentences.extend(p.strip() for p in parts if p.strip())

    # ── Step 3: greedily pack sentences into chunks ────────────────────────────
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        # If adding this sentence would exceed budget AND we already have content,
        # flush the current chunk first
        if current_len + sent_len + 1 > chunk_size and current_parts:
            chunks.append(" ".join(current_parts))
            # ── Step 4: overlap — seed next chunk with tail of current ─────────
            overlap_text = " ".join(current_parts)[-overlap:]
            # Walk forward to the first word boundary so we don't start mid-word
            space_idx = overlap_text.find(" ")
            overlap_seed = overlap_text[space_idx + 1:] if space_idx != -1 else overlap_text
            current_parts = [overlap_seed] if overlap_seed else []
            current_len = len(overlap_seed)

        current_parts.append(sent)
        current_len += sent_len + 1  # +1 for the joining space

    if current_parts:
        chunks.append(" ".join(current_parts))

    # ── Convert to extract_original()-compatible dicts ─────────────────────────
    # Each dict has the same keys that extract_original() returns so validate_chunk()
    # and the summarizer work without any changes.
    return [
        {
            "text":          chunk,
            "tables_html":   [],
            "images_base64": [],
            "table_count":   0,
            "image_count":   0,
        }
        for chunk in chunks
        if chunk.strip()
    ]


def chunk_by_title(elements):
    """
    Split on Title boundaries.
    Every chunk gets all elements between two titles.
    Tables and images that fall mid-section stay in that section's chunk
    and extract_original() will pull them out correctly.
    """
    chunks, current = [], []
    for el in elements:
        if el.category == "Title" and current:
            chunks.append(current)
            current = []
        current.append(el)
    if current:
        chunks.append(current)

    logger.info(f"chunk_by_title: {len(chunks)} chunks from {len(elements)} elements")
    return chunks


def extract_original(elements):
    """
    Extract text, tables (HTML), images (base64) from a chunk's element list.

    BUG FIXES vs original:
    1. Table: always append html (fallback to plain text if text_as_html missing)
               so tables_html is never empty when a Table element exists
    2. Image: only append base64 if it is not None/empty
               (hi_res without extract_image_block_to_payload gives None)
    3. Raw counts tracked separately from content lists
               so meta flags reflect actual element presence, not content quality
    """
    text          = []
    tables_html   = []
    images_base64 = []
    table_count   = 0
    image_count   = 0

    for el in elements:
        cat = el.category

        if cat == "Table":
            table_count += 1
            html = getattr(el.metadata, "text_as_html", None)
            if html:
                tables_html.append(html)
            else:
                # Fallback: wrap plain text in minimal HTML so it's not lost
                tables_html.append(f"<table><tr><td>{str(el)}</td></tr></table>")
            # Also add as text so table content is keyword-searchable
            text.append(str(el))

        elif cat in ("Image", "Figure"):
            image_count += 1
            b64 = getattr(el.metadata, "image_base64", None)
            if b64:  # only store if actually has content
                images_base64.append(b64)

        elif cat == "FigureCaption":
            # Captions describe figures — always keep for context
            text.append(str(el))

        else:
            text.append(str(el))

    result = {
        "text":          "\n".join(text).strip(),
        "tables_html":   tables_html,
        "images_base64": images_base64,
        # Raw element counts — used by pipeline.py for accurate meta flags
        "table_count":   table_count,
        "image_count":   image_count,
    }

    if table_count or image_count:
        logger.info(
            f"extract_original: {table_count} Table elements "
            f"({len(tables_html)} with HTML), "
            f"{image_count} Image elements "
            f"({len(images_base64)} with base64)"
        )

    return result


def validate_chunk(original: dict) -> bool:
    """
    Valid if has table/image content OR text meets minimum quality.
    Uses raw element counts so chunks with tables that lack HTML
    are still kept (not silently dropped).
    """
    # Use raw counts — not just content lists — so nothing gets dropped
    has_table = original.get("table_count", 0) > 0 or bool(original.get("tables_html"))
    has_image = original.get("image_count", 0) > 0 or bool(original.get("images_base64"))

    if has_table or has_image:
        return True

    text = original.get("text", "")
    if not text or len(text) < MIN_CHUNK_LENGTH:
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio >= MIN_ALPHA_RATIO
