import asyncio
import csv
import time
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import BATCH_SIZE
from src.graph import get_graph
from src.data_processing.formatting import format_choices_display, question_to_state
from src.data_processing.validators import normalize_answer
from src.utils.ingestion import ingest_all_data
from src.utils.logging import (
    log_done,
    log_pipeline,
    log_stats,
    print_log,
)
from src.data_processing.models import QuestionInput, PredictionOutput

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
            print(format_choices_display(q.choices))
            state = question_to_state(q)
            result = await graph.ainvoke(state)

            answer = result["answer"]
            route = result.get("route", "unknown")
            num_choices = len(q.choices)

            # Postprocess: validate and normalize answer
            normalized_answer = normalize_answer(
                answer=answer,
                num_choices=num_choices,
                question_id=q.qid,
                default="A",
            )

            log_done(f"{q.qid}: {normalized_answer} (Route: {route})")
            return PredictionOutput(qid=q.qid, answer=normalized_answer)

    tasks = [process_single_question(q) for q in questions]
    predictions = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time
    throughput = total / elapsed if elapsed > 0 else 0
    log_stats(f"Completed {total} questions in {elapsed:.2f}s ({throughput:.2f} req/s)")

    return predictions


def save_predictions(predictions: list[PredictionOutput], output_path: Path, ensure_dir: bool = True) -> None:
    """Save predictions to CSV file.
    
    Args:
        predictions: List of prediction outputs
        output_path: Path to output CSV file
        ensure_dir: If True, create parent directory if it doesn't exist
    """
    if ensure_dir:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        for pred in predictions:
            writer.writerow({"qid": pred.qid, "answer": pred.answer})
    log_pipeline(f"Predictions saved to: {output_path}")