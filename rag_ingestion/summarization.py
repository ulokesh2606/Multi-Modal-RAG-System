import asyncio
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import SUMMARY_MODEL

load_dotenv()
logger = logging.getLogger("ingestion.summarizer")


class Summarizer:
    def __init__(self):
        self.llm = ChatOpenAI(model=SUMMARY_MODEL, temperature=0)

    def _build_prompt(self, original: dict) -> str:
        parts = []
        if original["text"]:
            parts.append(f"TEXT CONTEXT (grounding only, do NOT rewrite):\n{original['text']}")
        if original["tables_html"]:
            parts.append(f"TABLE CONTEXT:\nSummarise {len(original['tables_html'])} table(s).")
        if original["images_base64"]:
            parts.append(f"IMAGE CONTEXT:\nDescribe {len(original['images_base64'])} image(s).")
        return "\n\n".join(parts)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def summarize(self, original: dict) -> str:
        if not (original["tables_html"] or original["images_base64"]):
            return original["text"]
        return self.llm.invoke(self._build_prompt(original)).content.strip()

    async def summarize_async(self, original: dict) -> str:
        return await asyncio.get_event_loop().run_in_executor(None, self.summarize, original)
