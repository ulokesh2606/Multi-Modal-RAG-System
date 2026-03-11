import math
import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI
from retrieval_config import (
    CONFIDENCE_HIGH_TOP_SCORE,
    CONFIDENCE_HIGH_AVG_SCORE,
    CONFIDENCE_HIGH_MAX_SPREAD,
    CONFIDENCE_MEDIUM_TOP_SCORE,
)

logger = logging.getLogger("rag.answer_gen")

ESCALATION_THRESHOLD = float(0.25)


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def analyze_scores(scored_docs: List[Dict]) -> Dict:
    if not scored_docs:
        return {
            "confidence":       "low",
            "top_score":        0.0,
            "avg_score":        0.0,
            "score_spread":     0.0,
            "top_score_norm":   0.0,
            "avg_score_norm":   0.0,
            "needs_escalation": True,
        }

    scores    = [d["score"] for d in scored_docs]
    top_score = scores[0]
    avg_score = sum(scores) / len(scores)
    spread    = max(scores) - min(scores)

    if top_score > CONFIDENCE_HIGH_TOP_SCORE:
        confidence = "high"
    elif avg_score > CONFIDENCE_HIGH_AVG_SCORE and spread < CONFIDENCE_HIGH_MAX_SPREAD:
        confidence = "high"
    elif top_score > CONFIDENCE_MEDIUM_TOP_SCORE:
        confidence = "medium"
    else:
        confidence = "low"

    top_score_norm   = _sigmoid(top_score)
    avg_score_norm   = _sigmoid(avg_score)
    needs_escalation = top_score_norm < ESCALATION_THRESHOLD

    return {
        "confidence":      confidence,
        "top_score":       round(top_score, 3),
        "avg_score":       round(avg_score, 3),
        "score_spread":    round(spread, 3),
        "top_score_norm":  round(top_score_norm, 3),
        "avg_score_norm":  round(avg_score_norm, 3),
        "needs_escalation": needs_escalation,
    }


class AnswerGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def generate(self, query: str, scored_docs: List[Dict]) -> Dict:
        analysis         = analyze_scores(scored_docs)
        confidence       = analysis["confidence"]
        needs_escalation = analysis["needs_escalation"]

        if not scored_docs:
            return {
                "answer":     "The information is not available in the provided documents.",
                "confidence": "low",
                "metrics": {
                    "top_score":       0.0,
                    "avg_score":       0.0,
                    "score_spread":    0.0,
                    "top_score_norm":  0.0,
                    "avg_score_norm":  0.0,
                    "num_chunks_used": 0,
                },
                "sources":  [],
                "escalate": True,
            }

        # Build context — top 3 chunks only (keeps TPM low, enough for 3-4 sentence answer)
        top_docs = scored_docs[:3]
        context  = "\n\n---\n\n".join(d["doc"].page_content for d in top_docs)

        if needs_escalation:
            strictness_instruction = (
                "The retrieved context may only be partially relevant. "
                "Answer using whatever information is present. "
                "Be transparent if the context does not fully address the question."
            )
        elif confidence == "high":
            strictness_instruction = (
                "The retrieved context is highly relevant. "
                "Answer directly and confidently using the context provided."
            )
        else:
            strictness_instruction = (
                "The retrieved context is relevant. "
                "Answer using the information present in the context. "
                "Combine information across chunks if needed."
            )

        prompt = f"""You are a concise customer service assistant. Answer using ONLY the information in the context below.

{strictness_instruction}

Rules:
- Answer in 3 to 4 sentences maximum. Be direct and clear.
- Do NOT use bullet points, numbered lists, or headers — plain prose only.
- Do NOT repeat the question or add unnecessary preamble.
- You MAY synthesize information across context chunks.
- You MUST NOT use any external knowledge.
- If the context does not contain enough information, say exactly:
  "The information is not available in the provided documents."

Context:
{context}

Question:
{query}

Answer (3-4 sentences max):"""

        answer = self.llm.invoke(prompt).content.strip()

        if needs_escalation and "not available" not in answer.lower():
            answer += (
                "\n\n_Note: My confidence in this answer is limited. "
                "If this is critical, please contact support for verification._"
            )

        logger.info(
            f"Answer generated ({confidence} confidence, "
            f"top_score={analysis['top_score']} → norm={analysis['top_score_norm']}, "
            f"escalate={needs_escalation}), "
            f"{len(top_docs)}/{len(scored_docs)} chunks used."
        )

        return {
            "answer":     answer,
            "confidence": confidence,
            "metrics": {
                "top_score":       analysis["top_score"],
                "avg_score":       analysis["avg_score"],
                "score_spread":    analysis["score_spread"],
                "top_score_norm":  analysis["top_score_norm"],
                "avg_score_norm":  analysis["avg_score_norm"],
                "num_chunks_used": len(top_docs),
            },
            "sources": [
                {
                    "chunk_id":   d["doc"].metadata.get("chunk_id"),
                    "score":      round(d["score"], 3),
                    "score_norm": round(_sigmoid(d["score"]), 3),
                }
                for d in top_docs
            ],
            "escalate": needs_escalation,
        }
