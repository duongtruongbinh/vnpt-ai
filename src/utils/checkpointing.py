"""Checkpointing utilities for resumable pipeline execution."""

import asyncio
import csv
import json
from pathlib import Path

from src.data_processing.models import InferenceLogEntry
from src.utils.common import sort_qids

_jsonl_lock = asyncio.Lock()


def load_log_entries(log_path: Path) -> dict[str, dict]:
    """Load all log entries from JSONL file as a dictionary.
    
    Args:
        log_path: Path to the JSONL log file
        
    Returns:
        Dictionary mapping qid to full entry data
    """
    entries = {}
    if not log_path.exists():
        return entries

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "qid" in entry:
                    entries[entry["qid"]] = entry
            except json.JSONDecodeError:
                continue

    return entries


def load_processed_qids(log_path: Path) -> set[str]:
    """Load already processed question IDs from JSONL log.
    
    Args:
        log_path: Path to the JSONL log file
        
    Returns:
        Set of question IDs that have been processed
    """
    return set(load_log_entries(log_path).keys())


async def append_log_entry(log_path: Path, entry: InferenceLogEntry) -> None:
    """Append a single log entry to JSONL file (thread-safe).
    
    Args:
        log_path: Path to the JSONL log file
        entry: InferenceLogEntry to append
    """
    async with _jsonl_lock:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")


def consolidate_log_file(log_path: Path) -> None:
    """Consolidate and sort log file by qid.
    
    Reads all entries, removes duplicates (keeps latest), and writes back sorted.
    
    Args:
        log_path: Path to the JSONL log file
    """
    if not log_path.exists():
        return
    
    entries = load_log_entries(log_path)
    if not entries:
        return
    
    sorted_qids = sort_qids(list(entries.keys()))
    
    # Write back sorted entries
    with open(log_path, "w", encoding="utf-8") as f:
        for qid in sorted_qids:
            f.write(json.dumps(entries[qid], ensure_ascii=False) + "\n")
    

def generate_csv_from_log(log_path: Path, output_path: Path) -> int:
    """Generate submission CSV from JSONL log, sorted by qid.
    
    Args:
        log_path: Path to the JSONL log file
        output_path: Path to the output CSV file
        
    Returns:
        Count of entries written to CSV
    """
    entries = load_log_entries(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_qids = sort_qids(list(entries.keys()))
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
        writer.writeheader()
        for qid in sorted_qids:
            writer.writerow({"qid": qid, "answer": entries[qid]["final_answer"]})

    return len(entries)


def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a VNPT API rate limit error."""
    
    error_str = str(error).lower()
    return (
        "429" in error_str or 
        "too many requests" in error_str or
        ("401" in error_str and "rate limit" in error_str) or
        "quota exceeded" in error_str
    )
