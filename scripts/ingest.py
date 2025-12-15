#!/usr/bin/env python
"""Build step script: Ingest documents into Qdrant vector database.

This is the dedicated "Build Step" script for the Pre-baked Database strategy.
Run this BEFORE building the Docker image to create the vector database.

Supports:
- Full knowledge base ingestion with ingest_all_data(force=True)
- Individual file ingestion (JSON, PDF, DOCX, TXT)
- Directory scanning
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import DATA_DIR, settings
from src.utils.ingestion import ingest_all_data, ingest_files

EPILOG = """
Examples:
  # Build complete knowledge base (recommended before Docker build)
  python scripts/ingest.py --build

  # Force rebuild entire knowledge base
  python scripts/ingest.py --build --force

  # Ingest specific files
  python scripts/ingest.py data/crawl/file.json
  python scripts/ingest.py data/crawl/*.json --append

  # Ingest directory
  python scripts/ingest.py --dir data/documents
"""


def main():
    parser = argparse.ArgumentParser(
        description="Build Step: Ingest documents into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("files", nargs="*", help="Files to ingest (JSON, PDF, DOCX, TXT)")
    parser.add_argument("--build", action="store_true", help="Build complete knowledge base from DATA_DIR")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion (wipe and rebuild)")
    parser.add_argument("--dir", help="Directory containing files to ingest")
    parser.add_argument("--collection", help=f"Collection name (default: {settings.qdrant_collection})")
    parser.add_argument("--append", action="store_true", help="Append to existing collection")
    
    args = parser.parse_args()
    
    # Build mode: full knowledge base ingestion
    if args.build:
        print(f"[Build] Building knowledge base from: {DATA_DIR}")
        print(f"[Build] Vector DB path: {settings.vector_db_path_resolved}")
        print(f"[Build] Force rebuild: {args.force}")
        print("-" * 50)
        
        try:
            ingest_all_data(base_dir=DATA_DIR, force=args.force)
            print("\n[Build] Knowledge base ready.")
            print(f"[Build] Output: {settings.vector_db_path_resolved}")
        except KeyboardInterrupt:
            print("\n[Cancelled]")
            sys.exit(1)
        except Exception as e:
            print(f"\n[Error] {e}")
            sys.exit(1)
        return
    
    # File mode: individual file ingestion
    file_paths = []
    
    if args.files:
        for f in args.files:
            path = Path(f)
            if path.exists():
                file_paths.append(path)
            else:
                print(f"[Warning] File not found: {f}")
    
    if args.dir:
        dir_path = Path(args.dir)
        if dir_path.is_dir():
            for ext in ["*.json", "*.pdf", "*.docx", "*.txt"]:
                file_paths.extend(dir_path.glob(ext))
        else:
            print(f"[Error] Directory not found: {args.dir}")
            sys.exit(1)
    
    if not file_paths:
        parser.print_help()
        print("\n[Error] No files specified. Use --build for full KB ingestion.")
        sys.exit(1)
    
    print(f"[Ingest] Files to process: {len(file_paths)}")
    print("-" * 40)
    
    try:
        ingest_files(file_paths, args.collection, args.append)
        print("\n[Done]")
    except KeyboardInterrupt:
        print("\n[Cancelled]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
