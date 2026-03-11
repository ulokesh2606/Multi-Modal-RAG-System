import sys
import logging

# Log to stderr — subprocess.run(capture_output=True) only captures stdout
# So these logs appear live in the WSL uvicorn terminal during every upload
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stderr
)

from pipeline import RAGIngestionPipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli.py <path> [tenant_id] [original_filename]")
        sys.exit(1)

    file_path         = sys.argv[1]
    tenant_id         = sys.argv[2] if len(sys.argv) > 2 else "default"
    original_filename = sys.argv[3] if len(sys.argv) > 3 else None

    pipeline = RAGIngestionPipeline()
    result   = pipeline.ingest(
        file_path,
        tenant_id=tenant_id,
        original_filename=original_filename
    )
    print(result)
