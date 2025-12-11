"""Core pipeline execution logic for the RAG system."""

import asyncio
import csv
import string
import sys
import time
from pathlib import Path

from src.config import BATCH_SIZE, DATA_OUTPUT_DIR
from src.data_processing.answer import normalize_answer
from src.data_processing.formatting import format_choices_display, question_to_state
from src.data_processing.models import InferenceLogEntry, PredictionOutput, QuestionInput
from src.graph import get_graph
from src.utils.checkpointing import (
    append_log_entry,
    consolidate_log_file,
    generate_csv_from_log,
    is_rate_limit_error,
)
from src.utils.common import sort_qids
from src.utils.ingestion import ingest_all_data
from src.utils.logging import log_done, log_pipeline, log_stats, print_log


def sort_questions_by_qid(questions: list[QuestionInput]) -> list[QuestionInput]:
    """Sort questions by qid using natural sorting."""
    qid_to_question = {q.qid: q for q in questions}
    sorted_qids = sort_qids(list(qid_to_question.keys()))
    return [qid_to_question[qid] for qid in sorted_qids]


async def run_pipeline_async(
    questions: list[QuestionInput],
    force_reingest: bool = False,
    batch_size: int = BATCH_SIZE,
) -> list[PredictionOutput]:
    """Run pipeline without checkpointing (for app.py deployment).

    Args:
        questions: List of questions to process
        force_reingest: If True, force re-ingestion of knowledge base
        batch_size: Number of concurrent questions to process

    Returns:
        List of PredictionOutput objects sorted by qid
    """
    log_pipeline("Initializing knowledge base...")
    ingest_all_data(force=force_reingest)

    questions = sort_questions_by_qid(questions)

    graph = get_graph()
    total = len(questions)
    start_time = time.perf_counter()

    sem = asyncio.Semaphore(batch_size)
    results: dict[str, PredictionOutput] = {}

    async def process_single_question(q: QuestionInput) -> None:
        async with sem:
            print_log(f"\n[{q.qid}] {q.question}")
            print_log(format_choices_display(q.choices))
            state = question_to_state(q)
            result = await graph.ainvoke(state)

            answer = result.get("answer", "A")
            route = result.get("route", "unknown")
            num_choices = len(q.choices)

            normalized_answer = normalize_answer(
                answer=answer,
                num_choices=num_choices,
                question_id=q.qid,
                default="A",
            )

            log_done(f"{q.qid}: {normalized_answer} (Route: {route})")
            results[q.qid] = PredictionOutput(qid=q.qid, answer=normalized_answer)

    tasks = [process_single_question(q) for q in questions]
    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time
    throughput = total / elapsed if elapsed > 0 else 0
    log_stats(f"Completed {total} questions in {elapsed:.2f}s ({throughput:.2f} req/s)")

    sorted_qids = sort_qids(list(results.keys()))
    return [results[qid] for qid in sorted_qids]


async def run_pipeline_with_checkpointing(
    questions: list[QuestionInput],
    log_path: Path,
    force_reingest: bool = False,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Run pipeline with JSONL checkpointing for resume capability.

    Questions are processed in qid order. Results are appended to log file
    immediately for fault tolerance, then consolidated at the end.

    Args:
        questions: List of questions to process (already filtered for unprocessed)
        log_path: Path to JSONL log file for checkpointing
        force_reingest: If True, force re-ingestion of knowledge base
        batch_size: Number of concurrent questions to process

    Returns:
        Count of newly processed questions
    """
    log_pipeline("Initializing knowledge base...")
    ingest_all_data(force=force_reingest)

    questions = sort_questions_by_qid(questions)
    log_pipeline(f"Processing {len(questions)} questions in qid order...")

    graph = get_graph()
    total = len(questions)
    start_time = time.perf_counter()
    processed_count = 0

    sem = asyncio.Semaphore(batch_size)
    stop_event = asyncio.Event()

    async def process_single_question(q: QuestionInput) -> None:
        nonlocal processed_count
        if stop_event.is_set():
            return

        async with sem:
            if stop_event.is_set():
                return
            print_log(f"\n[{q.qid}] {q.question}")
            print_log(format_choices_display(q.choices))
            state = question_to_state(q)

            try:
                result = await graph.ainvoke(state)
                answer = result.get("answer", "A")
                route = result.get("route", "unknown")
                raw_response = result.get("raw_response", "")
                context = result.get("context", "")

                num_choices = len(q.choices)
                option_labels = string.ascii_uppercase
                valid_answers = option_labels[:num_choices]

                if answer not in valid_answers:
                    print_log(f"        [Warning] Invalid answer '{answer}' for {q.qid}, defaulting to A")
                    answer = "A"

                log_entry = InferenceLogEntry(
                    qid=q.qid,
                    question=q.question,
                    choices=q.choices,
                    final_answer=answer,
                    raw_response=raw_response,
                    route=route,
                    retrieved_context=context,
                )
                await append_log_entry(log_path, log_entry)

                log_done(f"{q.qid}: {answer} (Route: {route})")
                processed_count += 1
                # await asyncio.sleep(35)

            except Exception as e:
                if is_rate_limit_error(e):
                    print_log(f"        [CRITICAL] Rate Limit Detected on {q.qid}: {e}")
                    stop_event.set()
                else:
                    print_log(f"        [Error] Failed to process {q.qid}: {e}")

    tasks = [asyncio.create_task(process_single_question(q)) for q in questions]
    await asyncio.gather(*tasks)

    if stop_event.is_set():
        log_pipeline("!!! PIPELINE STOPPED DUE TO RATE LIMIT !!!")
        log_pipeline("Consolidating logs and generating emergency submission...")
        consolidate_log_file(log_path)

        output_file = DATA_OUTPUT_DIR / "submission_emergency.csv"
        total_entries = generate_csv_from_log(log_path, output_file)
        log_pipeline(f"Saved emergency submission with {total_entries} entries to: {output_file}")

        sys.exit(0)

    log_pipeline("Consolidating log file...")
    consolidate_log_file(log_path)

    elapsed = time.perf_counter() - start_time
    throughput = total / elapsed if elapsed > 0 else 0
    log_stats(f"Processed {processed_count}/{total} questions in {elapsed:.2f}s ({throughput:.2f} req/s)")

    return processed_count


def save_predictions(
    predictions: list[PredictionOutput],
    output_path: Path,
    ensure_dir: bool = True,
) -> None:
    """Save predictions to CSV file, sorted by qid.

    Args:
        predictions: List of prediction outputs
        output_path: Path to output CSV file
        ensure_dir: If True, create parent directory if it doesn't exist
    """
    if ensure_dir:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_qids = sort_qids([p.qid for p in predictions])
    pred_dict = {p.qid: p for p in predictions}

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        for qid in sorted_qids:
            writer.writerow({"qid": qid, "answer": pred_dict[qid].answer})
    log_pipeline(f"Predictions saved to: {output_path}")
