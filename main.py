import asyncio
from pathlib import Path

from src.config import BATCH_SIZE, DATA_INPUT_DIR, DATA_OUTPUT_DIR
from src.pipeline import run_pipeline_async, save_predictions
from src.data_processing.loaders import load_test_data_from_json
from src.utils.llm import get_large_model, get_small_model
from src.utils.logging import log_main


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
    questions = load_test_data_from_json(input_file)
    log_main(f"Loaded {len(questions)} questions (batch_size={batch_size})")

    predictions = await run_pipeline_async(questions, batch_size=batch_size)

    output_file = DATA_OUTPUT_DIR / "submission.csv"
    save_predictions(predictions, output_file, ensure_dir=False)


def main(batch_size: int = BATCH_SIZE) -> None:
    """Main entry point that runs the async pipeline."""
    asyncio.run(async_main(batch_size=batch_size))


if __name__ == "__main__":
    main()