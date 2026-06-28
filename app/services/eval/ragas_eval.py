"""
RAGAS evaluation harness.

Offline CLI tool — not a live API endpoint.
Measures retrieval quality across 4 RAGAS metrics.

Usage:
    python -m app.services.eval.ragas_eval \
        --project "Project Atlas" \
        --output results/ragas_report.json

Metrics evaluated:
    - faithfulness:        Are answers grounded in retrieved context?
    - answer_relevancy:    Does the answer address the question?
    - context_precision:   Are retrieved chunks actually relevant?
    - context_recall:      Did retrieval find all relevant chunks?
"""

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.search import SearchQuery
from app.services.agent.analysis_agent import AnalysisAgent
from app.services.retrieval.reranker import MsMarcoReranker
from app.services.retrieval.vector_store import HybridRetriever
from app.db.qdrant_client import QdrantClientFactory
from app.services.ingestion.embedder import DocumentEmbedder

logger = logging.getLogger(__name__)


# ── Evaluation Test Set ────────────────────────────────────────────────────────
# [WHY] Hardcoded eval set for MVP — these are the ground truth
# question/answer pairs we use to measure retrieval quality.
# Sprint 2: load from a CSV/JSON file for easier maintenance.

EVAL_QUESTIONS = [
    {
        "question": "Which suppliers are causing delays in Project Atlas?",
        "ground_truth": (
            "BearingTech GmbH (Supplier X) is causing a 26-day delay "
            "in Project Atlas due to a quality hold on bearing shipment "
            "Part #BT-4492. The revised delivery date is April 8, 2024."
        ),
        "project_name": "Project Atlas",
    },
    {
        "question": "What procurement risks are affecting turbine delivery?",
        "ground_truth": (
            "The turbine delivery milestone is delayed due to BearingTech GmbH "
            "failing quality inspection on 8 of 24 bearings. "
            "The financial exposure is EUR 240,000 in penalty clauses."
        ),
        "project_name": "Project Atlas",
    },
    {
        "question": "What is blocking the commissioning phase of Project Atlas?",
        "ground_truth": (
            "Commissioning is blocked by a chain of dependencies: "
            "bearing delivery delay → turbine assembly delay → "
            "electrical wiring delay → commissioning delay. "
            "The minimum additional delay is 26 days."
        ),
        "project_name": "Project Atlas",
    },
    {
        "question": "What quality issues are affecting Project Titan?",
        "ground_truth": (
            "The inlet manifold from Precision Parts Co (Part #PP-7721) "
            "failed dimensional inspection with 1.8mm tolerance variance. "
            "NCR #2024-031 has been issued. FastParts International "
            "can supply a replacement in 10 days at 40% premium."
        ),
        "project_name": "Project Titan",
    },
    {
        "question": "Which milestones are at risk in Project Titan?",
        "ground_truth": (
            "Compressor Installation is BLOCKED pending replacement part. "
            "Pressure Testing is AT RISK due to dependency on compressor. "
            "Control Panel Approval is PENDING with client for 17 days "
            "blocking switchgear order and electrical installation."
        ),
        "project_name": "Project Titan",
    },
]


async def _build_retriever(settings) -> HybridRetriever:
    from langchain_huggingface import HuggingFaceEmbeddings

    factory = QdrantClientFactory(settings=settings)
    await factory.connect()

    # [WHY] ragas_eval runs as standalone CLI process.
    # Must build its own embeddings instance —
    # cannot share the app singleton from dependencies.py
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    embedder = DocumentEmbedder(embeddings=embeddings)  # ← correct signature
    reranker = MsMarcoReranker()

    return HybridRetriever(
        client=factory.get_client(),
        collection_name=settings.qdrant_collection_name,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
    )


async def _run_eval(output_path: Path) -> dict:
    """
    Run RAGAS evaluation over the test question set.

    Args:
        output_path: Path to write JSON results.

    Returns:
        Dict of metric scores.
    """
    settings = get_settings()
    retriever = await _build_retriever()
    agent = AnalysisAgent(retriever=retriever, settings=settings)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    logger.info(
        "Running eval over question set",
        extra={"question_count": len(EVAL_QUESTIONS)},
    )

    for item in EVAL_QUESTIONS:
        query = SearchQuery(
            query=item["question"],
            project_name=item["project_name"],
            top_k=10,
            rerank=True,
        )

        # ── Retrieve context ───────────────────────────────────────────────
        try:
            results = await retriever.search(query)
            context_texts = [r.content for r in results]
        except Exception as exc:
            logger.error(
                "Retrieval failed for eval question",
                extra={"question": item["question"], "error": str(exc)},
            )
            context_texts = []

        # ── Generate answer via agent ──────────────────────────────────────
        try:
            report = await agent.run(
                query=item["question"],
                project_name=item["project_name"],
            )
            answer = report.summary
        except Exception as exc:
            logger.error(
                "Agent failed for eval question",
                extra={"question": item["question"], "error": str(exc)},
            )
            answer = "Agent failed to generate answer."

        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(item["ground_truth"])

        logger.info(
            "Eval question processed",
            extra={
                "question": item["question"][:60],
                "context_chunks": len(context_texts),
            },
        )

    # ── RAGAS Dataset ──────────────────────────────────────────────────────
    samples = [
    SingleTurnSample(
        user_input=q,
        response=a,
        retrieved_contexts=c,
        reference=g,
    )
    for q, a, c, g in zip(questions, answers, contexts, ground_truths)
]
    eval_dataset = EvaluationDataset(samples=samples)

    # ── RAGAS LLM + Embeddings ─────────────────────────────────────────────
    # [WHY] RAGAS uses its own LLM and embeddings for metric computation.
    # We pass our Azure instances so RAGAS does not try to use OpenAI directly.
    ragas_llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        max_tokens=2048,
    )

    ragas_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Run RAGAS ─────────────────────────────────────────────────────────
    logger.info("Running RAGAS metric computation...")

    try:
        result = evaluate(
        dataset=eval_dataset,
        metrics=[
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ],
)
    except Exception as exc:
        logger.error("RAGAS evaluation failed", extra={"error": str(exc)})
        raise

    scores = {
    "faithfulness": round(float(result.scores["faithfulness"]), 4),
    "answer_relevancy": round(float(result.scores["answer_relevancy"]), 4),
    "context_precision": round(float(result.scores["context_precision"]), 4),
    "context_recall": round(float(result.scores["context_recall"]), 4),
    "evaluated_at": datetime.now(UTC).isoformat(),
    "question_count": len(EVAL_QUESTIONS),
}

    # ── Write results ──────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2))

    logger.info("RAGAS evaluation complete", extra=scores)
    return scores


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on Manufacturing Risk Navigator"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ragas_report.json"),
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Filter eval to specific project (optional)",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    scores = asyncio.run(_run_eval(output_path=args.output))

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in scores.items():
        if isinstance(score, float):
            bar = "█" * int(score * 20)
            print(f"{metric:<25} {score:.4f}  {bar}")
    print("=" * 50)
    print(f"Full report → {args.output}")


if __name__ == "__main__":
    main()