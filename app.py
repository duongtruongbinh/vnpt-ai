import asyncio
from pathlib import Path

from src.config import BATCH_SIZE, DATA_INPUT_DIR, DATA_OUTPUT_DIR
from src.pipeline import run_pipeline_async, save_predictions
from src.data_processing.loaders import load_test_data_from_csv
from src.utils.llm import get_large_model, get_small_model
from src.utils.logging import log_main


def _find_test_file() -> Path | None:
    """Find the first available test CSV file in DATA_INPUT_DIR."""
    candidates = [
        "public_test.csv",
        "private_test.csv",
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
    log_main("Models warmed up ready.")

    input_file = _find_test_file()

    if input_file is None:
        raise FileNotFoundError(
            f"No test CSV file found in {DATA_INPUT_DIR}. "
            "Expected files: public_test.csv or private_test.csv"
        )

    log_main(f"Loading test data from: {input_file}")
    questions = load_test_data_from_csv(input_file)
    log_main(f"Loaded {len(questions)} questions (batch_size={batch_size})")

    predictions = await run_pipeline_async(questions, batch_size=batch_size)

    output_file = DATA_OUTPUT_DIR / "pred.csv"
    save_predictions(predictions, output_file, ensure_dir=True)
    log_main(f"Predictions saved to: {output_file}")


def main(batch_size: int = BATCH_SIZE) -> None:
    """Main entry point that runs the async pipeline."""
    asyncio.run(async_main(batch_size=batch_size))


if __name__ == "__main__":
    main()