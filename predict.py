import asyncio
import csv
import json
from pathlib import Path

from src.config import BATCH_SIZE
from src.data_processing.loaders import load_test_data_from_json 
from src.pipeline import run_pipeline_async
from src.utils.llm import get_large_model, get_small_model
from src.utils.logging import log_main

INPUT_FILE = Path("/code/private_test.json")
OUTPUT_FILE = Path("submission.csv")

async def async_main(batch_size: int = BATCH_SIZE) -> None:

    get_small_model()
    get_large_model()
    log_main("Models warmed up ready.")

    if not INPUT_FILE.exists():
        local_test = Path("data/test.json") 
        if local_test.exists():
            log_main(f"Warning: {INPUT_FILE} not found. Using local test file: {local_test}")
            input_path = local_test
        else:
            raise FileNotFoundError(f"Input file not found at {INPUT_FILE}")
    else:
        input_path = INPUT_FILE

    log_main(f"Loading test data from: {input_path}")
    questions = load_test_data_from_json(input_path)
    log_main(f"Loaded {len(questions)} questions")

    predictions = await run_pipeline_async(questions, batch_size=batch_size)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for p in predictions:
            writer.writerow([p.qid, p.answer])
            
    log_main(f"Predictions saved to: {OUTPUT_FILE.absolute()}")

def main():
    asyncio.run(async_main(batch_size=BATCH_SIZE))

if __name__ == "__main__":
    main()