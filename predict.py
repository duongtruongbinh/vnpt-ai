import asyncio
import os
from pathlib import Path

from src.config import BATCH_SIZE, DATA_INPUT_DIR, DATA_OUTPUT_DIR
from src.pipeline import run_pipeline_async, save_predictions
from src.data_processing.loaders import load_test_data_from_csv, load_test_data_from_json
from src.utils.llm import get_large_model, get_small_model
from src.utils.logging import log_main


def _find_test_file() -> Path | None:
    """
    Find the input file.
    Priority:
    1. Environment Variable INPUT_FILE_PATH (Defined in Dockerfile)
    2. private_test.json (Standard submission file)
    3. public_test.json (For testing)
    4. Fallback to CSV if JSON not found
    """
    env_path = os.getenv("INPUT_FILE_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        log_main(f"[WARNING] INPUT_FILE_PATH defined as {env_path} but file does not exist. Searching default paths...")

    candidates = [
        "private_test.json",
        "public_test.json",
        "test.json",
        "private_test.csv",
        "public_test.csv",
    ]
    
    for filename in candidates:
        path = DATA_INPUT_DIR / filename
        if path.exists():
            return path
            
    return None


async def async_main(batch_size: int = BATCH_SIZE) -> None:
    """Async main entry point for deployment."""
    get_small_model()
    get_large_model()
    log_main("Models/API warmed up ready.")

    input_file = _find_test_file()

    if input_file is None:
        try:
            files_in_data = list(DATA_INPUT_DIR.glob("*"))
            files_str = ", ".join([f.name for f in files_in_data])
        except Exception:
            files_str = "Cannot list directory"
            
        raise FileNotFoundError(
            f"No suitable test file found in {DATA_INPUT_DIR}. "
            f"Searched for private_test.json/csv. Found files: {files_str}"
        )

    log_main(f"Loading test data from: {input_file}")
    
    if input_file.suffix.lower() == ".json":
        questions = load_test_data_from_json(input_file)
    elif input_file.suffix.lower() == ".csv":
        questions = load_test_data_from_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
        
    log_main(f"Loaded {len(questions)} questions")

    predictions = await run_pipeline_async(questions, batch_size=batch_size)

    output_file = DATA_OUTPUT_DIR / "submission.csv"
    save_predictions(predictions, output_file, ensure_dir=True)
    log_main(f"Predictions saved to: {output_file}")


def main(batch_size: int = BATCH_SIZE) -> None:
    """Main entry point that runs the async pipeline."""
    asyncio.run(async_main(batch_size=batch_size))


if __name__ == "__main__":
    main()