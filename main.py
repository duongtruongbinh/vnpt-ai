"""Entry point for local testing/development with resume capability."""

import asyncio
from pathlib import Path

from src.config import BATCH_SIZE, DATA_INPUT_DIR, DATA_OUTPUT_DIR
from src.data_processing.loaders import load_test_data_from_json
from src.pipeline import run_pipeline_with_checkpointing
from src.utils.checkpointing import (
    consolidate_log_file,
    generate_csv_from_log,
    load_processed_qids,
)
from src.utils.common import sort_qids
from src.utils.llm import get_large_model, get_small_model
from src.utils.logging import log_main, log_pipeline


def _find_test_file() -> Path | None:
    """Find the first available test JSON file in DATA_INPUT_DIR."""
    candidates = [
        "test.json",
        "val.json",
        "private_test.json",
        "public_test.json",
    ]
    for filename in candidates:
        path = DATA_INPUT_DIR / filename
        if path.exists():
            return path
    return None


async def async_main(batch_size: int = BATCH_SIZE) -> None:
    """Async main entry point with resume capability."""
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
    all_questions = load_test_data_from_json(input_file)
    log_main(f"Loaded {len(all_questions)} total questions")

    log_path = DATA_OUTPUT_DIR / "inference_log.jsonl"
    processed_qids = load_processed_qids(log_path)

    if processed_qids:
        log_main(f"Resuming: Found {len(processed_qids)} already processed questions")
        
        remaining_questions = [q for q in all_questions if q.qid not in processed_qids]
        
        remaining_qids = sort_qids([q.qid for q in remaining_questions])
        qid_to_q = {q.qid: q for q in remaining_questions}
        remaining_questions = [qid_to_q[qid] for qid in remaining_qids]
        
        if remaining_questions:
            log_main(f"Remaining: {len(remaining_questions)} questions to process")
            first_qid = remaining_questions[0].qid
            last_qid = remaining_questions[-1].qid
            log_main(f"Processing qid range: {first_qid} -> {last_qid}")
        else:
            log_main("All questions already processed. Skipping inference.")
    else:
        log_main("Starting fresh run (no existing checkpoint found)")
        remaining_questions = all_questions

    if remaining_questions:
        await run_pipeline_with_checkpointing(
            remaining_questions,
            log_path,
            batch_size=batch_size,
        )
    
    log_pipeline("Final consolidation: sorting log file by qid...")
    consolidate_log_file(log_path)
    
    output_file = DATA_OUTPUT_DIR / "submission.csv"
    total_entries = generate_csv_from_log(log_path, output_file)
    log_pipeline(f"Generated submission.csv with {total_entries} entries (sorted by qid): {output_file}")
    
    all_qids = set(q.qid for q in all_questions)
    processed_now = load_processed_qids(log_path)
    missing = all_qids - processed_now
    if missing:
        missing_sorted = sort_qids(list(missing))
        log_main(f"Warning: {len(missing)} questions still missing: {missing_sorted[:5]}{'...' if len(missing) > 5 else ''}")
    else:
        log_main(f"All {len(all_qids)} questions have been processed!")


def main(batch_size: int = BATCH_SIZE) -> None:
    """Main entry point that runs the async pipeline."""
    asyncio.run(async_main(batch_size=batch_size))


if __name__ == "__main__":
    main()
