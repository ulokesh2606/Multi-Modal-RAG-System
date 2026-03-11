from typing import List
from langchain_openai import ChatOpenAI


class QueryExpander:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def generate(self, user_query: str) -> List[str]:
        prompt = f"""You are generating search queries for a RAG system.

Original question:
"{user_query}"

Generate exactly 3 alternative queries that:
- Use different technical phrasing
- Expand acronyms if any
- Emphasize definitions, mechanisms, or steps

Return one query per line. Do not explain."""

        response = self.llm.invoke(prompt).content.strip()
        variations = [q.strip() for q in response.split("\n") if q.strip()]
        return [user_query] + variations[:3]

