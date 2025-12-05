"""Entry point for running the RAG pipeline on test data."""

import asyncio
import csv
import time
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import DATA_INPUT_DIR, DATA_OUTPUT_DIR, BATCH_SIZE
from src.graph import GraphState, get_graph
from src.utils.ingestion import ingest_knowledge_base
from src.utils.llm import get_small_model, get_large_model


class QuestionInput(BaseModel):
    """Input schema for a multiple-choice question."""

    id: str = Field(description="Question identifier")
    question: str = Field(description="Question text in Vietnamese")
    A: str = Field(description="Option A")
    B: str = Field(description="Option B")
    C: str = Field(description="Option C")
    D: str = Field(description="Option D")
    category: str | None = Field(default=None, description="Question category")


class PredictionOutput(BaseModel):
    """Output schema for a prediction."""

    id: str = Field(description="Question identifier")
    answer: str = Field(description="Predicted answer: A, B, C, or D")


def load_test_data(file_path: Path) -> list[QuestionInput]:
    """Load test questions from CSV file."""
    questions = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(QuestionInput(**row))
    return questions


def question_to_state(q: QuestionInput) -> GraphState:
    """Convert QuestionInput to GraphState."""
    return {
        "question_id": q.id,
        "question": q.question,
        "option_a": q.A,
        "option_b": q.B,
        "option_c": q.C,
        "option_d": q.D,
    }


def result_to_prediction(result: dict, question_id: str) -> PredictionOutput:
    """Convert graph result to PredictionOutput."""
    answer = result.get("answer", "A")
    if answer not in ["A", "B", "C", "D"]:
        answer = "A"
    return PredictionOutput(id=question_id, answer=answer)


async def run_pipeline_async(
    questions: list[QuestionInput],
    force_reingest: bool = False,
    batch_size: int = BATCH_SIZE,
) -> list[PredictionOutput]:
    """Run pipeline with Semaphore for maximum throughput."""
    
    print("[Pipeline] Initializing knowledge base...")
    ingest_knowledge_base(force=force_reingest)

    graph = get_graph()
    total = len(questions)
    start_time = time.perf_counter()
    
    sem = asyncio.Semaphore(batch_size)

    async def process_single_question(q: QuestionInput):
        async with sem:
            state = question_to_state(q)
            result = await graph.ainvoke(state)
            
            answer = result.get("answer", "A")
            if answer not in ["A", "B", "C", "D"]:
                answer = "A"
            
            route = result.get("route", "unknown")
            print(f"  [Done] {q.id}: {answer} (Route: {route})")
            return PredictionOutput(id=q.id, answer=answer)

    print(f"[Pipeline] Processing {total} questions with concurrency limit = {batch_size}...")
    
    tasks = [process_single_question(q) for q in questions]
    
    predictions = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time
    throughput = total / elapsed if elapsed > 0 else 0
    print(f"\n[Pipeline] Completed {total} questions in {elapsed:.2f}s "
          f"({throughput:.2f} req/s)")

    return predictions


def save_predictions(predictions: list[PredictionOutput], output_path: Path) -> None:
    """Save predictions to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        for pred in predictions:
            writer.writerow({"qid": pred.id, "answer": pred.answer})
    print(f"[Pipeline] Predictions saved to: {output_path}")


async def async_main(batch_size: int = BATCH_SIZE) -> None:
    """Async main entry point."""
    get_small_model()
    get_large_model()
    print("[Main] Models warmed up ready.")
    
    input_file = DATA_INPUT_DIR / "private_test.csv"
    if not input_file.exists():
        input_file = DATA_INPUT_DIR / "public_test.csv"

    if not input_file.exists():
        print("[Main] Test file not found. Generating dummy data...")
        from scripts.generate_data import generate_knowledge_base
        generate_knowledge_base()

    print(f"[Main] Loading test data from: {input_file}")
    questions = load_test_data(input_file)
    print(f"[Main] Loaded {len(questions)} questions (batch_size={batch_size})")

    predictions = await run_pipeline_async(questions, batch_size=batch_size)

    output_file = DATA_OUTPUT_DIR / "pred.csv"
    save_predictions(predictions, output_file)

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for pred in predictions:
        print(f"  {pred.id}: {pred.answer}")


def main(batch_size: int = BATCH_SIZE) -> None:
    """Main entry point that runs the async pipeline."""
    asyncio.run(async_main(batch_size=batch_size))


if __name__ == "__main__":
    main()
