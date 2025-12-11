from pydantic import BaseModel, Field


class QuestionInput(BaseModel):
    """Input schema for a multiple-choice question."""

    qid: str = Field(description="Question identifier")
    question: str = Field(description="Question text in Vietnamese")
    choices: list[str] = Field(description="List of answer choices")
    answer: str | None = Field(default=None, description="Correct answer (A, B, C, ...)")


class PredictionOutput(BaseModel):
    """Output schema for a prediction."""

    qid: str = Field(description="Question identifier")
    answer: str = Field(description="Predicted answer: A, B, C, D, ...")


class InferenceLogEntry(BaseModel):
    """Schema for JSONL inference log entry (used for checkpointing)."""

    qid: str = Field(description="Question identifier")
    question: str = Field(description="Original question text")
    choices: list[str] = Field(description="List of answer choices")
    final_answer: str = Field(description="Final predicted answer")
    raw_response: str = Field(default="", description="Raw LLM response")
    route: str = Field(default="unknown", description="Pipeline route taken")
    retrieved_context: str = Field(default="", description="Retrieved context from RAG")