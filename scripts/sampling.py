"""
sampling.py

Builds a frozen, stratified random sample of 200 sentences per dataset
(CoNLL-2014, BEA-2019, JFLEG), split into three equal-sized length groups
(short / medium / long) using a fixed random seed for full reproducibility.

Output: a single CSV (sampled_sentences.csv) that becomes the canonical
input for every GEC system in the pipeline (GECToR, LanguageTool,
Grammarly, ChatGPT, Claude, DeepSeek). Every system runs against exactly
these same sentences.
"""

import random
import csv
from pathlib import Path

RANDOM_SEED = 2026
SENTENCES_PER_DATASET = 200
N_STRATA = 3  # short, medium, long

random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# 1. Dataset-specific parsers
#    Each returns a list of dicts: {"original_id": ..., "sentence": ...}
# ---------------------------------------------------------------------

def parse_conll2014(m2_path):
    """
    Parses the CoNLL-2014 combined .m2 file and extracts source sentences.
    Source sentences are lines starting with 'S '.
    """
    sentences = []
    with open(m2_path, encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                text = line[2:].strip()
                sentences.append({
                    "original_id": f"conll14_{idx}",
                    "sentence": text,
                })
                idx += 1
    return sentences


def parse_bea2019(m2_path):
    """
    Parses a BEA-2019 (W&I+LOCNESS) .m2 file (e.g. the dev set, which
    ships with gold annotations) and extracts source sentences.
    Identical M2 format to CoNLL-2014, so the same parser logic applies.
    """
    sentences = []
    with open(m2_path, encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                text = line[2:].strip()
                sentences.append({
                    "original_id": f"bea19_{idx}",
                    "sentence": text,
                })
                idx += 1
    return sentences


def parse_jfleg(src_path):
    """
    Parses a JFLEG .src file (plain text, one sentence per line).
    """
    sentences = []
    with open(src_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            text = line.strip()
            if text:
                sentences.append({
                    "original_id": f"jfleg_{idx}",
                    "sentence": text,
                })
    return sentences


# ---------------------------------------------------------------------
# 2. Stratified sampling
# ---------------------------------------------------------------------

def word_count(sentence):
    return len(sentence.split())


def assign_length_bucket(sentences):
    """
    Splits sentences into 3 equal-sized buckets (short/medium/long) based
    on word-count terciles computed from this dataset's own distribution.
    """
    sorted_by_len = sorted(sentences, key=lambda s: word_count(s["sentence"]))
    n = len(sorted_by_len)
    third = n // 3

    for i, s in enumerate(sorted_by_len):
        s["word_count"] = word_count(s["sentence"])
        if i < third:
            s["length_bucket"] = "short"
        elif i < 2 * third:
            s["length_bucket"] = "medium"
        else:
            s["length_bucket"] = "long"
    return sorted_by_len


def stratified_sample(sentences, dataset_name, total=SENTENCES_PER_DATASET):
    """
    Draws an equal number of sentences from each length bucket (total
    split as evenly as possible across 3 strata) using the module-level
    fixed random seed.
    """
    bucketed = assign_length_bucket(sentences)

    by_bucket = {"short": [], "medium": [], "long": []}
    for s in bucketed:
        by_bucket[s["length_bucket"]].append(s)

    base = total // N_STRATA
    remainder = total % N_STRATA
    sizes = [base + (1 if i < remainder else 0) for i in range(N_STRATA)]

    sample = []
    for bucket_name, size in zip(["short", "medium", "long"], sizes):
        available = by_bucket[bucket_name]
        if len(available) < size:
            raise ValueError(
                f"{dataset_name}: not enough '{bucket_name}' sentences "
                f"({len(available)} available, {size} needed). "
                f"Check the input file or reduce SENTENCES_PER_DATASET."
            )
        sample.extend(random.sample(available, size))

    for s in sample:
        s["dataset"] = dataset_name

    return sample


# ---------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------

def main():
    # ---- Verified against your actual uploaded files ----
    conll_path = Path("conll14st-test-data/noalt/official-2014.combined.m2")
    bea_path = Path("wi+locness/m2/ABCN.dev.gold.bea19.m2")
    jfleg_path = Path("jfleg-master/test/test.src")
    # -------------------------------------------------------

    datasets = {
        "conll14": parse_conll2014(conll_path),
        "bea19": parse_bea2019(bea_path),
        "jfleg": parse_jfleg(jfleg_path),
    }

    all_samples = []
    for name, sentences in datasets.items():
        print(f"{name}: {len(sentences)} sentences available")
        sample = stratified_sample(sentences, name)
        n_short = sum(1 for s in sample if s["length_bucket"] == "short")
        n_med = sum(1 for s in sample if s["length_bucket"] == "medium")
        n_long = sum(1 for s in sample if s["length_bucket"] == "long")
        print(f"{name}: sampled {len(sample)} sentences "
              f"({n_short} short, {n_med} medium, {n_long} long)")
        all_samples.extend(sample)

    out_path = Path("sampled_sentences.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "original_id", "sentence", "word_count", "length_bucket"],
        )
        writer.writeheader()
        for s in all_samples:
            writer.writerow({
                "dataset": s["dataset"],
                "original_id": s["original_id"],
                "sentence": s["sentence"],
                "word_count": s["word_count"],
                "length_bucket": s["length_bucket"],
            })

    print(f"\nSaved {len(all_samples)} total sentences to {out_path}")


if __name__ == "__main__":
    main()
