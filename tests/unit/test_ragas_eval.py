import sys
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── THE IMPORT BYPASS ────────────────────────────────────────────────────────
# Inject mock modules into sys.modules before importing your application code.
# This completely prevents the `ModuleNotFoundError` during pytest collection.
sys.modules["ragas"] = MagicMock()
sys.modules["ragas.metrics"] = MagicMock()

from app.services.eval import ragas_eval


def test_eval_questions_have_required_fields():
    """Every eval question must have question, ground_truth, project_name."""
    for item in ragas_eval.EVAL_QUESTIONS:
        assert "question" in item, f"Missing question: {item}"
        assert "ground_truth" in item, f"Missing ground_truth: {item}"
        assert "project_name" in item, f"Missing project_name: {item}"
        assert len(item["question"]) > 10
        assert len(item["ground_truth"]) > 10


def test_eval_questions_cover_multiple_projects():
    """Eval set must cover more than one project."""
    projects = {item["project_name"] for item in ragas_eval.EVAL_QUESTIONS}
    assert len(projects) >= 2


def test_eval_question_count():
    """Eval set must have at least 5 questions."""
    assert len(ragas_eval.EVAL_QUESTIONS) >= 5


@pytest.mark.asyncio
async def test_run_eval_writes_output_file(tmp_path):
    """Eval run must write a JSON file with expected metric keys."""
    output = tmp_path / "ragas_report.json"

    # Mock the RAGAS evaluate() output
    mock_ragas_result = {
        "faithfulness": 0.87,
        "answer_relevancy": 0.91,
        "context_precision": 0.83,
        "context_recall": 0.78,
    }

    # ── THE LOGIC FIX ─────────────────────────────────────────────────────────
    # We patch the external dependencies (the retriever, agent, and RAGAS evaluate)
    # but we CALL the actual _run_eval function so it writes the file itself.
    with patch("app.services.eval.ragas_eval._build_retriever", new_callable=AsyncMock), \
         patch("app.services.eval.ragas_eval.AnalysisAgent") as MockAgent, \
         patch("app.services.eval.ragas_eval.evaluate", return_value=mock_ragas_result):

        # Setup the mock agent to return a fake generated answer
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run = AsyncMock(return_value=MagicMock(summary="Mock Answer"))

        # Execute the ACTUAL function
        await ragas_eval._run_eval(output_path=output)

    # Assert the function created the file
    assert output.exists(), "The _run_eval function failed to create the output file."

    # Read the file and verify its contents
    data = json.loads(output.read_text())

    assert "faithfulness" in data
    assert "evaluated_at" in data
    assert data["question_count"] == len(ragas_eval.EVAL_QUESTIONS)
    assert 0.0 <= data["faithfulness"] <= 1.0