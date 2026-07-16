"""
run_gector.py  —  run INSIDE the GECToR Docker container.

Reads sampled_sentences.csv from the mounted /app/data volume,
runs every sentence through GECToR once (deterministic, no repetitions),
and appends results to outputs.csv in the same volume.

Usage (from Windows terminal):
    docker run --rm \
        -v D:/Mestrado/docker/gector/data:/app/data \
        -v D:/Mestrado/pipeline:/app/pipeline \
        gector-env \
        python3 /app/pipeline/run_gector.py

Resume-safe: rows already present in outputs.csv are skipped.
"""

import csv
import sys
import time
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer
from gector import GECToR, predict, load_verb_dict

# ── paths (inside the container) ────────────────────────────────────────
DATA_DIR        = Path("/app/data")
INPUT_CSV       = DATA_DIR / "sampled_sentences.csv"
OUTPUT_CSV      = DATA_DIR / "outputs.csv"

MODEL_PATH      = DATA_DIR / "gector-2024-roberta-large.th"
VOCAB_PATH      = DATA_DIR / "output_vocabulary"
VERB_VOCAB_PATH = DATA_DIR / "verb-form-vocab.txt"

# ── output schema (shared with pipeline.py) ──────────────────────────────
OUTPUT_FIELDS = [
    "original_id", "dataset", "length_bucket", "word_count",
    "sentence", "system", "model_version", "run_index",
    "corrected", "error", "timestamp",
]

SYSTEM_NAME    = "gector"
MODEL_VERSION  = "gector-2024-roberta-large"
RUN_INDEX      = 1  # deterministic — always 1


def load_model():
    """Load GECToR model with CPU-mapping patch."""
    _orig = torch.load
    def _cpu_load(*a, **kw):
        kw.setdefault("map_location", torch.device("cpu"))
        return _orig(*a, **kw)
    torch.load = _cpu_load

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GECToR.from_official_pretrained(
        str(MODEL_PATH),
        special_tokens_fix=1,
        transformer_model="roberta-large",
        vocab_path=str(VOCAB_PATH),
        max_length=80,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=True)
    encode, decode = load_verb_dict(str(VERB_VOCAB_PATH))
    print(f"[gector] Model loaded on {device}", flush=True)
    return model, tokenizer, encode, decode


def already_done(output_csv):
    """Return set of original_ids already written for this system."""
    done = set()
    if output_csv.exists():
        with open(output_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("system") == SYSTEM_NAME:
                    done.add(row["original_id"])
    return done


def main():
    if not INPUT_CSV.exists():
        print(f"[gector] ERROR: {INPUT_CSV} not found. Mount the pipeline volume correctly.")
        sys.exit(1)

    # Read all sentences
    with open(INPUT_CSV, encoding="utf-8") as f:
        sentences = list(csv.DictReader(f))
    print(f"[gector] {len(sentences)} sentences in input CSV", flush=True)

    done = already_done(OUTPUT_CSV)
    todo = [s for s in sentences if s["original_id"] not in done]
    print(f"[gector] {len(done)} already done, {len(todo)} to process", flush=True)

    if not todo:
        print("[gector] Nothing to do — all sentences already logged.")
        return

    model, tokenizer, encode, decode = load_model()

    write_header = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(todo, start=1):
            sentence = row["sentence"]
            error = ""
            corrected = ""
            try:
                result = predict(
                    model, tokenizer, [sentence], encode, decode,
                    keep_confidence=0.0,
                    min_error_prob=0.0,
                    n_iteration=5,
                    batch_size=1,
                )
                corrected = result[0]
            except Exception as e:
                error = str(e)
                print(f"[gector] ERROR on {row['original_id']}: {e}", flush=True)

            writer.writerow({
                "original_id":   row["original_id"],
                "dataset":       row["dataset"],
                "length_bucket": row["length_bucket"],
                "word_count":    row["word_count"],
                "sentence":      sentence,
                "system":        SYSTEM_NAME,
                "model_version": MODEL_VERSION,
                "run_index":     RUN_INDEX,
                "corrected":     corrected,
                "error":         error,
                "timestamp":     datetime.utcnow().isoformat(),
            })
            f.flush()  # write immediately so progress survives interruption

            if i % 50 == 0 or i == len(todo):
                print(f"[gector] {i}/{len(todo)} processed", flush=True)

    print(f"[gector] Done. Results appended to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
