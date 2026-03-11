from datetime import datetime, timezone


def start_report(document_id: str, filename: str) -> dict:
    return {
        "document_id": document_id,
        "file": filename,
        "stats": {
            "atomic_elements": 0,
            "chunks_created": 0,
            "chunks_embedded": 0,
            "chunks_deduped": 0,
            "tables_detected": 0,
            "images_detected": 0
        },
        "processing_time_sec": 0,
        "completed_at": None
    }


def finalize_report(report: dict, duration: float):
    report["processing_time_sec"] = round(duration, 2)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()

