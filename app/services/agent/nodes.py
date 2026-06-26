"""
LangGraph node functions.

Each node:
  - Receives AgentState
  - Does one thing
  - Returns partial AgentState update
"""

import json
import logging
import uuid
from typing import Any

import tiktoken
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

from app.core.config import AppSettings
from app.core.exceptions import AgentError, ContextWindowError
from app.models.analysis import DependencyEdge, RiskItem
from app.models.observability import TokenUsage
from app.models.search import SearchQuery, SearchResult
from app.models.timeline import Milestone
from app.services.agent.prompts import (
    DEPENDENCY_EXTRACTION_PROMPT,
    RISK_IDENTIFICATION_PROMPT,
    SUMMARISE_PROMPT,
    TIMELINE_INFERENCE_PROMPT,
)
from app.services.agent.state import AgentState
from app.services.retrieval.vector_store import HybridRetriever

logger = logging.getLogger(__name__)

# [WHY] Token budget for context passed to LLM nodes.
# GPT-4o-mini = 128k context. We reserve 4k for prompt templates
# and 2k for completion. Remaining ~10k for chunks is conservative
# but keeps costs predictable.
_MAX_CONTEXT_TOKENS = 10_000
_TOKENISER = tiktoken.get_encoding("cl100k_base")


def _build_context(chunks: list[SearchResult]) -> str:
    """
    Format retrieved chunks into LLM-readable context string.

    Each chunk includes its chunk_id for citation tracing.
    """
    lines = []
    for chunk in chunks:
        lines.append(
            f"[chunk_id: {chunk.chunk_id}]\n"
            f"Source: {chunk.metadata.file_name} "
            f"(page {chunk.metadata.page_number})\n"
            f"{chunk.content}\n"
            f"{'─' * 40}"
        )
    return "\n".join(lines)


def _count_tokens(text: str) -> int:
    """Count tokens in a string using cl100k_base tokeniser."""
    return len(_TOKENISER.encode(text))


def _trim_chunks_to_budget(
    chunks: list[SearchResult],
    budget: int,
) -> tuple[list[SearchResult], bool]:
    """
    Trim chunk list to fit within token budget.

    Removes lowest-scoring chunks first (they are at the end
    after hybrid search sorts by final_score descending).

    Returns:
        (trimmed_chunks, was_truncated)
    """
    total = 0
    kept: list[SearchResult] = []

    for chunk in chunks:
        chunk_tokens = _count_tokens(chunk.content)
        if total + chunk_tokens > budget:
            logger.warning(
                "Token budget reached — trimming remaining chunks",
                extra={
                    "kept": len(kept),
                    "dropped": len(chunks) - len(kept),
                    "budget": budget,
                },
            )
            return kept, True
        total += chunk_tokens
        kept.append(chunk)

    return kept, False


def _accumulate_tokens(
    existing: TokenUsage,
    prompt: int,
    completion: int,
) -> TokenUsage:
    """Merge new token counts into running total."""
    return TokenUsage(
        prompt_tokens=existing.prompt_tokens + prompt,
        completion_tokens=existing.completion_tokens + completion,
        total_tokens=existing.total_tokens + prompt + completion,
    )


async def _call_llm_json(
    llm: BaseChatModel,
    prompt: str,
) -> tuple[Any, int, int]:
    """
    Call LLM and parse JSON response.

    Returns:
        (parsed_json, prompt_tokens, completion_tokens)

    Raises:
        AgentError: If LLM call or JSON parsing fails.
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        usage = response.response_metadata.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
    except Exception as exc:
        raise AgentError(
            message="LLM call failed",
            detail=str(exc),
        ) from exc

    try:
        # [WHY] Strip markdown fences — LLM sometimes wraps JSON
        # in ```json ... ``` despite instructions.
        clean = content.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise AgentError(
            message="LLM returned invalid JSON",
            detail=f"{str(exc)} | raw={content[:200]}",
        ) from exc

    return parsed, prompt_tokens, completion_tokens


# ── Node Functions ─────────────────────────────────────────────────────────────

async def retrieve_node(
    state: AgentState,
    retriever: HybridRetriever,
    settings: AppSettings,
) -> dict:
    """
    Node 1: Retrieve relevant chunks via hybrid search.

    Applies token budget guard before passing chunks forward.
    Returns empty chunks (not error) if nothing found.
    """
    logger.info(
        "Agent node: retrieve",
        extra={"query": state["query"], "project": state["project_name"]},
    )

    query = SearchQuery(
        query=state["query"],
        project_name=state["project_name"],
        top_k=20,
        rerank=True,
    )

    try:
        results = await retriever.search(query)
    except Exception as exc:
        logger.error("Retrieval failed in agent", extra={"error": str(exc)})
        results = []

    if not results:
        logger.warning("No chunks retrieved — agent will return empty report")
        return {
            "retrieved_chunks": [],
            "context_truncated": False,
        }

    trimmed, truncated = _trim_chunks_to_budget(results, _MAX_CONTEXT_TOKENS)

    return {
        "retrieved_chunks": trimmed,
        "context_truncated": truncated,
    }


async def risk_node(
    state: AgentState,
    llm: BaseChatModel,
) -> dict:
    """
    Node 2a (parallel): Identify risks from retrieved chunks.

    Returns empty list if no chunks available.
    """
    logger.info("Agent node: risk_identification")

    if not state["retrieved_chunks"]:
        return {"risks": [], "token_usage": state["token_usage"]}

    context = _build_context(state["retrieved_chunks"])
    prompt = RISK_IDENTIFICATION_PROMPT.format(
        project_name=state["project_name"],
        query=state["query"],
        context=context,
    )

    raw, prompt_tokens, completion_tokens = await _call_llm_json(llm, prompt)

    risks: list[RiskItem] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            # [WHY] Ensure risk_id exists — LLM sometimes omits it.
            item.setdefault("risk_id", str(uuid.uuid4()))
            item.setdefault("detected_at", None)
            risks.append(RiskItem(**item))
        except Exception as exc:
            logger.warning(
                "Skipping malformed risk item",
                extra={"error": str(exc), "item": str(item)[:100]},
            )

    logger.info("Risk node complete", extra={"risks_found": len(risks)})

    return {
        "risks": risks,
        "token_usage": _accumulate_tokens(
            state["token_usage"], prompt_tokens, completion_tokens
        ),
    }


async def dependency_node(
    state: AgentState,
    llm: BaseChatModel,
) -> dict:
    """
    Node 2b (parallel): Extract dependency edges from retrieved chunks.

    Returns empty list if no chunks available.
    """
    logger.info("Agent node: dependency_extraction")

    if not state["retrieved_chunks"]:
        return {"dependencies": [], "token_usage": state["token_usage"]}

    context = _build_context(state["retrieved_chunks"])
    prompt = DEPENDENCY_EXTRACTION_PROMPT.format(
        project_name=state["project_name"],
        query=state["query"],
        context=context,
    )

    raw, prompt_tokens, completion_tokens = await _call_llm_json(llm, prompt)

    dependencies: list[DependencyEdge] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            dependencies.append(DependencyEdge(**item))
        except Exception as exc:
            logger.warning(
                "Skipping malformed dependency item",
                extra={"error": str(exc)},
            )

    logger.info(
        "Dependency node complete",
        extra={"dependencies_found": len(dependencies)},
    )

    return {
        "dependencies": dependencies,
        "token_usage": _accumulate_tokens(
            state["token_usage"], prompt_tokens, completion_tokens
        ),
    }


async def timeline_node(
    state: AgentState,
    llm: BaseChatModel,
) -> dict:
    """
    Node 3: Infer milestone status using risks + dependencies.

    Runs after both parallel nodes complete.
    """
    logger.info("Agent node: timeline_inference")

    if not state["retrieved_chunks"]:
        return {"milestones": [], "token_usage": state["token_usage"]}

    context = _build_context(state["retrieved_chunks"])
    prompt = TIMELINE_INFERENCE_PROMPT.format(
        project_name=state["project_name"],
        query=state["query"],
        context=context,
        risks=json.dumps(
            [r.model_dump(mode="json") for r in state["risks"]],
            indent=2,
        ),
        dependencies=json.dumps(
            [d.model_dump(mode="json") for d in state["dependencies"]],
            indent=2,
        ),
    )

    raw, prompt_tokens, completion_tokens = await _call_llm_json(llm, prompt)

    milestones: list[Milestone] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            item.setdefault("milestone_id", str(uuid.uuid4()))
            milestones.append(Milestone(**item))
        except Exception as exc:
            logger.warning(
                "Skipping malformed milestone",
                extra={"error": str(exc)},
            )

    logger.info(
        "Timeline node complete",
        extra={"milestones_found": len(milestones)},
    )

    return {
        "milestones": milestones,
        "token_usage": _accumulate_tokens(
            state["token_usage"], prompt_tokens, completion_tokens
        ),
    }


async def summarise_node(
    state: AgentState,
    llm: BaseChatModel,
) -> dict:
    """
    Node 4: Generate grounded executive summary narrative.

    Final node before END. Always produces a summary string.
    """
    logger.info("Agent node: summarise")

    if not state["retrieved_chunks"]:
        return {
            "summary": "No relevant project data found for this query.",
            "token_usage": state["token_usage"],
            "error": None,
        }

    prompt = SUMMARISE_PROMPT.format(
        project_name=state["project_name"],
        query=state["query"],
        risks=json.dumps(
            [r.model_dump(mode="json") for r in state["risks"]],
            indent=2,
        ),
        dependencies=json.dumps(
            [d.model_dump(mode="json") for d in state["dependencies"]],
            indent=2,
        ),
        milestones=json.dumps(
            [m.model_dump(mode="json") for m in state["milestones"]],
            indent=2,
        ),
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        usage = response.response_metadata.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
    except Exception as exc:
        raise AgentError(
            message="Summarise node LLM call failed",
            detail=str(exc),
        ) from exc

    logger.info("Summarise node complete")

    return {
        "summary": summary,
        "token_usage": _accumulate_tokens(
            state["token_usage"], prompt_tokens, completion_tokens
        ),
        "error": None,
    }