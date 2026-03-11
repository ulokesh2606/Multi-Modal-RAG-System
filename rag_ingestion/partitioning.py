import os
import logging
from collections import defaultdict

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from unstructured.partition.html import partition_html

logger = logging.getLogger("rag.partitioning")


def partition_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"PDF size: {file_size_mb:.1f} MB — running hi_res (yolox)")
        logger.info("This takes 3-10 minutes for academic PDFs. Please wait...")

        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            hi_res_model_name="yolox",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
        )

    elif ext == ".docx":
        elements = partition_docx(filename=file_path)
    elif ext == ".pptx":
        elements = partition_pptx(filename=file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            elements = partition_text(text=f.read())
    elif ext in (".html", ".htm"):
        elements = partition_html(filename=file_path)
    elif ext == ".csv":
        import pandas as pd
        elements = partition_text(text=pd.read_csv(file_path).to_markdown(index=False))
    elif ext == ".xlsx":
        import pandas as pd
        elements = partition_text(text=pd.read_excel(file_path).to_markdown(index=False))
    elif ext == ".md":
        with open(file_path, "r", encoding="utf-8") as f:
            elements = partition_text(text=f.read())
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    stats = defaultdict(int)
    for el in elements:
        stats[el.category] += 1

    logger.info(
        f"Partitioning done: {len(elements)} elements | "
        f"Tables={stats.get('Table', 0)} | "
        f"Images={stats.get('Image', 0) + stats.get('Figure', 0)} | "
        f"NarrativeText={stats.get('NarrativeText', 0)}"
    )
    return elements, dict(stats)
