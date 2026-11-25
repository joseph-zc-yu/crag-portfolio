import re
from dataclasses import dataclass, field
import pymupdf

HEADING_RE = re.compile(
    r"^(chapter|section|article|part|\d+(\.\d+)*\.?)\s+[A-Z0-9].{1,80}$",
    re.IGNORECASE,
)


@dataclass
class DocumentChunk:
    chunk_id: str
    section_title: str
    page_number: int  # 1-indexed primary page
    text: str
    # Maps page_number -> list of normalized [x0, y0, x1, y1] boxes
    bboxes_by_page: dict[int, list[list[float]]] = field(default_factory=dict)


def extract_section_chunks(pdf_bytes: bytes, max_chars: int = 1200) -> list[DocumentChunk]:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    toc = doc.get_toc()  # [[level, title, page_num], ...]
    toc_by_page: dict[int, str] = {}
    for _, title, page_num in toc:
        toc_by_page.setdefault(page_num, title.strip())

    chunks: list[DocumentChunk] = []
    current_section = "Introduction / Document Start"
    buf_text: list[str] = []
    buf_bboxes: dict[int, list[list[float]]] = {}
    buf_primary_page = 1

    def flush_buffer():
        nonlocal buf_text, buf_bboxes, buf_primary_page
        joined = "\n".join(buf_text).strip()
        if joined:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"c_{len(chunks) + 1}",
                    section_title=current_section,
                    page_number=buf_primary_page,
                    text=joined,
                    bboxes_by_page=buf_bboxes,
                )
            )
        buf_text = []
        buf_bboxes = {}

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        pw, ph = page.rect.width, page.rect.height
        if pw == 0 or ph == 0:
            continue

        if page_num in toc_by_page:
            # Start a fresh section if TOC marks a new heading on this page
            if sum(len(t) for t in buf_text) > 150:
                flush_buffer()
                buf_primary_page = page_num
            current_section = toc_by_page[page_num]

        blocks = page.get_text("blocks", sort=True)
        for b in blocks:
            x0, y0, x1, y1, text, _, block_type = b
            if block_type != 0:  # text blocks only
                continue
            clean = " ".join(text.split())
            if not clean:
                continue

            norm_box = [
                round(x0 / pw, 4),
                round(y0 / ph, 4),
                round(x1 / pw, 4),
                round(y1 / ph, 4),
            ]

            # Detect structural headings (Chapter/Section/Article)
            if HEADING_RE.match(clean) and len(clean) < 100:
                flush_buffer()
                current_section = clean
                buf_primary_page = page_num

            if not buf_text:
                buf_primary_page = page_num

            buf_text.append(clean)
            buf_bboxes.setdefault(page_num, []).append(norm_box)

            if sum(len(t) for t in buf_text) >= max_chars:
                flush_buffer()
                buf_primary_page = page_num

    flush_buffer()
    doc.close()
    return chunks