"""
pipeline.py  —  run on your WINDOWS HOST (not inside Docker).

Reads sampled_sentences.csv, runs each sentence through:
  - LanguageTool  (local Docker server, 1 run — deterministic)
  - ChatGPT       (OpenAI API,          3 runs — stochastic)
  - Claude        (Anthropic API,       3 runs — stochastic)
  - DeepSeek      (DeepSeek API,        3 runs — stochastic)

Appends all results to outputs.csv (same file run_gector.py writes to).
Resume-safe: already-logged (system, original_id, run_index) combos
are skipped, so you can stop and restart at any point.

Prerequisites:
  - LanguageTool container running:
      docker start languagetool
  - API keys set as environment variables:
      set OPENAI_API_KEY=sk-...
      set ANTHROPIC_API_KEY=sk-ant-...
      set DEEPSEEK_API_KEY=sk-...
  - pip install openai anthropic requests

Usage:
  python pipeline.py

To run only specific systems (e.g. while testing):
  python pipeline.py --systems languagetool chatgpt
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
# Edit INPUT_CSV / OUTPUT_CSV to wherever your pipeline folder lives.
INPUT_CSV  = Path("sampled_sentences.csv")
OUTPUT_CSV = Path("outputs.csv")

# ── output schema (shared with run_gector.py) ────────────────────────────
OUTPUT_FIELDS = [
    "original_id", "dataset", "length_bucket", "word_count",
    "sentence", "system", "model_version", "run_index",
    "corrected", "error", "timestamp",
]

# ── LanguageTool settings ────────────────────────────────────────────────
LT_ENDPOINT = "http://localhost:8010/v2/check"
LT_LANGUAGE = "en-US"

# ── LLM settings (imported from llm_wrappers.py) ─────────────────────────
# CORRECTION_PROMPT and model strings live there; keep them in sync.
LLM_RUNS = 3  # repetitions per sentence per stochastic system

# ── retry settings ───────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ────────────────────────────────────────────────────────────────────────
# LanguageTool correction
# ────────────────────────────────────────────────────────────────────────

def languagetool_correct(sentence):
    """
    Calls the local LanguageTool server and applies corrections
    right-to-left to avoid offset shifts.
    Returns the corrected sentence string.
    """
    import requests
    response = requests.post(
        LT_ENDPOINT,
        data={"text": sentence, "language": LT_LANGUAGE},
        timeout=30,
    )
    response.raise_for_status()
    matches = response.json().get("matches", [])
    corrected = sentence
    for match in sorted(matches, key=lambda m: m["offset"], reverse=True):
        replacements = match.get("replacements", [])
        if not replacements:
            continue
        start = match["offset"]
        end = start + match["length"]
        corrected = corrected[:start] + replacements[0]["value"] + corrected[end:]
    return corrected


# ────────────────────────────────────────────────────────────────────────
# LLM wrappers (imported from llm_wrappers.py)
# ────────────────────────────────────────────────────────────────────────

def _load_llm_wrappers():
    """
    Imports llm_wrappers.py from the same directory as this script.
    Deferred import so missing libraries only error when actually needed.
    """
    import importlib.util
    here = Path(__file__).parent / "llm_wrappers.py"
    spec = importlib.util.spec_from_file_location("llm_wrappers", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ────────────────────────────────────────────────────────────────────────
# Resume helpers
# ────────────────────────────────────────────────────────────────────────

def already_done(output_csv):
    """
    Returns a set of (system, original_id, run_index) tuples already
    present in the output CSV, so we can skip them on resume.
    """
    done = set()
    if output_csv.exists():
        with open(output_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["system"], row["original_id"], row["run_index"]))
    return done


# ────────────────────────────────────────────────────────────────────────
# Per-system runners
# ────────────────────────────────────────────────────────────────────────

def run_languagetool(sentences, done, writer, fh):
    total = len(sentences)
    processed = 0
    for row in sentences:
        key = ("languagetool", row["original_id"], "1")
        if key in done:
            continue
        error = ""
        corrected = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                corrected = languagetool_correct(row["sentence"])
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    error = str(e)
                    print(f"  [languagetool] ERROR on {row['original_id']}: {e}")
        writer.writerow({
            "original_id":   row["original_id"],
            "dataset":       row["dataset"],
            "length_bucket": row["length_bucket"],
            "word_count":    row["word_count"],
            "sentence":      row["sentence"],
            "system":        "languagetool",
            "model_version": "languagetool-6.8",
            "run_index":     1,
            "corrected":     corrected,
            "error":         error,
            "timestamp":     datetime.utcnow().isoformat(),
        })
        fh.flush()
        processed += 1
        if processed % 50 == 0 or processed == total:
            print(f"  [languagetool] {processed}/{total}")


def run_llm(system_name, correct_fn, model_version, sentences, done, writer, fh):
    total = len(sentences) * LLM_RUNS
    processed = 0
    for row in sentences:
        for run in range(1, LLM_RUNS + 1):
            key = (system_name, row["original_id"], str(run))
            if key in done:
                processed += 1
                continue
            error = ""
            corrected = ""
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = correct_fn(row["sentence"], run_index=run)
                    corrected = result["corrected"]
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY * attempt)
                    else:
                        error = str(e)
                        print(f"  [{system_name}] ERROR on {row['original_id']} run {run}: {e}")
            writer.writerow({
                "original_id":   row["original_id"],
                "dataset":       row["dataset"],
                "length_bucket": row["length_bucket"],
                "word_count":    row["word_count"],
                "sentence":      row["sentence"],
                "system":        system_name,
                "model_version": model_version,
                "run_index":     run,
                "corrected":     corrected,
                "error":         error,
                "timestamp":     datetime.utcnow().isoformat(),
            })
            fh.flush()
            processed += 1
            if processed % 50 == 0 or processed == total:
                print(f"  [{system_name}] {processed}/{total}")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEC pipeline runner")
    parser.add_argument(
        "--systems", nargs="+",
        choices=["languagetool", "chatgpt", "claude", "deepseek"],
        default=["languagetool", "chatgpt", "claude", "deepseek"],
        help="Which systems to run (default: all four)",
    )
    args = parser.parse_args()

    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run sampling.py first.")
        sys.exit(1)

    with open(INPUT_CSV, encoding="utf-8") as f:
        sentences = list(csv.DictReader(f))
    print(f"Loaded {len(sentences)} sentences from {INPUT_CSV}")

    done = already_done(OUTPUT_CSV)
    print(f"Resuming: {len(done)} (system, sentence, run) combos already logged\n")

    # Load LLM wrappers only if we'll need them
    llm = None
    if any(s in args.systems for s in ["chatgpt", "claude", "deepseek"]):
        llm = _load_llm_wrappers()

    write_header = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()

        if "languagetool" in args.systems:
            print("=== LanguageTool ===")
            run_languagetool(sentences, done, writer, fh)

        if "chatgpt" in args.systems:
            if not os.environ.get("OPENAI_API_KEY"):
                print("=== ChatGPT: SKIPPED (OPENAI_API_KEY not set) ===")
            else:
                print("=== ChatGPT ===")
                run_llm("chatgpt", llm.chatgpt_correct, llm.OPENAI_MODEL,
                        sentences, done, writer, fh)

        if "claude" in args.systems:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("=== Claude: SKIPPED (ANTHROPIC_API_KEY not set) ===")
            else:
                print("=== Claude ===")
                run_llm("claude", llm.claude_correct, llm.ANTHROPIC_MODEL,
                        sentences, done, writer, fh)

        if "deepseek" in args.systems:
            if not os.environ.get("DEEPSEEK_API_KEY"):
                print("=== DeepSeek: SKIPPED (DEEPSEEK_API_KEY not set) ===")
            else:
                print("=== DeepSeek ===")
                run_llm("deepseek", llm.deepseek_correct, llm.DEEPSEEK_MODEL,
                        sentences, done, writer, fh)

    print(f"\nAll done. Results in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
