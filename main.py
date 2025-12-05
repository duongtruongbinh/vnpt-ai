"""Entry point for running the RAG pipeline on test data."""

import asyncio
import csv
import json
import string
import time
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import BATCH_SIZE, DATA_INPUT_DIR, DATA_OUTPUT_DIR
from src.graph import get_graph
from src.state import GraphState
from src.utils.ingestion import ingest_all_data
from src.utils.llm import get_large_model, get_small_model
from src.utils.logging import (
    log_done,
    log_main,
    log_pipeline,
    log_stats,
    print_header,
    print_log,
)


class QuestionInput(BaseModel):
    """Input schema for a multiple-choice question."""

    qid: str = Field(description="Question identifier")
    question: str = Field(description="Question text in Vietnamese")
    choices: list[str] = Field(description="List of answer choices")
    answer: str | None = Field(default=None, description="Correct answer (A, B, C, ...)")


class PredictionOutput(BaseModel):
    """Output schema for a prediction."""

    qid: str = Field(description="Question identifier")
    answer: str = Field(description="Predicted answer: A, B, C, D, ... or 'Từ chối trả lời'")


def _choices_to_options(choices: list[str]) -> dict[str, str]:
    """Convert choices list to option dictionary (A, B, C, D, ...)."""
    option_labels = string.ascii_uppercase
    options = {}
    for i, choice in enumerate(choices):
        if i < len(option_labels):
            options[option_labels[i]] = choice
    return options


def load_test_data(file_path: Path) -> list[QuestionInput]:
    """Load test questions from JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    if file_path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON files are supported: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of questions: {file_path}")

    questions = []
    for item in data:
        if "choices" not in item or not isinstance(item["choices"], list):
            raise ValueError(f"Question {item.get('qid', 'unknown')} must have 'choices' as a list")

        questions.append(QuestionInput(
            qid=item["qid"],
            question=item["question"],
            choices=item["choices"],
            answer=item.get("answer"),
        ))

    return questions


def question_to_state(q: QuestionInput) -> GraphState:
    """Convert QuestionInput to GraphState."""
    options = _choices_to_options(q.choices)

    state: GraphState = {
        "question_id": q.qid,
        "question": q.question,
        "option_a": options.get("A", ""),
        "option_b": options.get("B", ""),
        "option_c": options.get("C", ""),
        "option_d": options.get("D", ""),
        "all_choices": q.choices,
    }
    return state


def _format_choices_display(choices: list[str]) -> str:
    """Format choices for display."""
    option_labels = string.ascii_uppercase
    lines = []
    for i in range(0, len(choices), 2):
        line_parts = []
        for j in range(2):
            idx = i + j
            if idx < len(choices):
                label = option_labels[idx] if idx < len(option_labels) else str(idx)
                line_parts.append(f"{label}. {choices[idx]:<30}")
        if line_parts:
            lines.append("   " + " ".join(line_parts))
    return "\n".join(lines)


async def run_pipeline_async(
    questions: list[QuestionInput],
    force_reingest: bool = False,
    batch_size: int = BATCH_SIZE,
) -> list[PredictionOutput]:
    """Run pipeline with Semaphore for maximum throughput."""
    log_pipeline("Initializing knowledge base...")
    ingest_all_data(force=force_reingest)

    graph = get_graph()
    total = len(questions)
    start_time = time.perf_counter()

    sem = asyncio.Semaphore(batch_size)

    async def process_single_question(q: QuestionInput):
        async with sem:
            print_log(f"\n[{q.qid}] {q.question}")
            print(_format_choices_display(q.choices))
            state = question_to_state(q)
            result = await graph.ainvoke(state)

            answer = result["answer"]
            route = result.get("route", "unknown")

            num_choices = len(q.choices)
            option_labels = string.ascii_uppercase
            valid_answers = option_labels[:num_choices]

            if answer not in valid_answers:
                if answer.upper() in ["TỪ CHỐI TRẢ LỜI", "TỪCHỐITRẢLỜI"]:
                    answer = "Từ chối trả lời"
                else:
                    print_log(f"        [Warning] Invalid answer '{answer}' for {q.qid}, defaulting to A")
                    answer = "A"

            log_done(f"{q.qid}: {answer} (Route: {route})")
            return PredictionOutput(qid=q.qid, answer=answer)

    tasks = [process_single_question(q) for q in questions]
    predictions = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time
    throughput = total / elapsed if elapsed > 0 else 0
    log_stats(f"Completed {total} questions in {elapsed:.2f}s ({throughput:.2f} req/s)")

    return predictions


def save_predictions(predictions: list[PredictionOutput], output_path: Path) -> None:
    """Save predictions to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        for pred in predictions:
            writer.writerow({"qid": pred.qid, "answer": pred.answer})
    log_pipeline(f"Predictions saved to: {output_path}")


def _find_test_file() -> Path | None:
    """Find the first available test JSON file in DATA_INPUT_DIR."""
    candidates = [
        "val.json",
        "test.json",
        "private_test.json",
        "public_test.json",
    ]
    for filename in candidates:
        path = DATA_INPUT_DIR / filename
        if path.exists():
            return path
    return None


async def async_main(batch_size: int = BATCH_SIZE) -> None:
    """Async main entry point."""
    get_small_model()
    get_large_model()
    log_main("Models warmed up ready.")

    input_file = _find_test_file()

    if input_file is None:
        raise FileNotFoundError(
            f"No test JSON file found in {DATA_INPUT_DIR}. "
            "Expected files: test.json, val.json, private_test.json, or public_test.json"
        )

    log_main(f"Loading test data from: {input_file}")
    questions = load_test_data(input_file)
    log_main(f"Loaded {len(questions)} questions (batch_size={batch_size})")

    predictions = await run_pipeline_async(questions, batch_size=batch_size)

    output_file = DATA_OUTPUT_DIR / "submission.csv"
    save_predictions(predictions, output_file)


def main(batch_size: int = BATCH_SIZE) -> None:
    """Main entry point that runs the async pipeline."""
    asyncio.run(async_main(batch_size=batch_size))


if __name__ == "__main__":
    main()
