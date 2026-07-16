"""
llm_wrappers.py

API wrapper functions for the three stochastic LLM systems:
ChatGPT (OpenAI), Claude (Anthropic), and DeepSeek.

Each function takes a sentence and a run index (1-3, since each sentence
is processed 3 times per stochastic system) and returns a dict consistent
with the LanguageTool wrapper shape, for uniform downstream CSV logging.

To change the prompt for all systems at once, edit CORRECTION_PROMPT only.
"""

import os
import time

# -----------------------------------------------------------------------
# THE PROMPT — edit this one constant to change it for all three systems.
# -----------------------------------------------------------------------
CORRECTION_PROMPT = (
    "Correct the grammatical errors in the following sentence "
    "and return only the corrected sentence with no explanation: {sentence}"
)

# Model version strings — pinned explicitly for reproducibility.
# Update these if you switch models, and document the change.
OPENAI_MODEL   = "gpt-4o"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEEPSEEK_MODEL  = "deepseek-chat"  # DeepSeek V3 (standard, not reasoning)

# Retry settings for transient API errors (rate limits, timeouts)
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _build_prompt(sentence):
    """Fills the shared prompt template with the given sentence."""
    return CORRECTION_PROMPT.format(sentence=sentence)


def _extract_correction(raw_text):
    """
    Strips common LLM response artifacts (leading/trailing whitespace,
    quote wrapping) from the raw model output to get a clean sentence.
    """
    text = raw_text.strip()
    # Some models wrap the output in quotes — strip them if present
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1].strip()
    return text


# -----------------------------------------------------------------------
# ChatGPT (OpenAI)
# -----------------------------------------------------------------------

def chatgpt_correct(sentence, run_index=1):
    """
    Sends a sentence to the OpenAI API and returns a corrected version.

    Args:
        sentence:   The source sentence to correct.
        run_index:  Which repetition this is (1, 2, or 3). Logged in the
                    output dict but does not affect the API call itself —
                    stochasticity comes from the model, not the caller.

    Returns:
        dict with keys: original, corrected, model, run_index, system
    """
    import openai  # pip install openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = _build_prompt(sentence)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,   # default; keep consistent across runs
                max_tokens=256,
            )
            raw = response.choices[0].message.content
            return {
                "system": "chatgpt",
                "model": OPENAI_MODEL,
                "run_index": run_index,
                "original": sentence,
                "corrected": _extract_correction(raw),
            }
        except openai.RateLimitError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
        except openai.APIError as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


# -----------------------------------------------------------------------
# Claude (Anthropic)
# -----------------------------------------------------------------------

def claude_correct(sentence, run_index=1):
    """
    Sends a sentence to the Anthropic API and returns a corrected version.

    Args:
        sentence:   The source sentence to correct.
        run_index:  Which repetition this is (1, 2, or 3).

    Returns:
        dict with keys: original, corrected, model, run_index, system
    """
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = _build_prompt(sentence)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            return {
                "system": "claude",
                "model": ANTHROPIC_MODEL,
                "run_index": run_index,
                "original": sentence,
                "corrected": _extract_correction(raw),
            }
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
        except anthropic.APIError as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


# -----------------------------------------------------------------------
# DeepSeek
# -----------------------------------------------------------------------

def deepseek_correct(sentence, run_index=1):
    """
    Sends a sentence to the DeepSeek API and returns a corrected version.
    DeepSeek's API is OpenAI-compatible, so we use the openai client
    pointed at DeepSeek's base URL.

    Args:
        sentence:   The source sentence to correct.
        run_index:  Which repetition this is (1, 2, or 3).

    Returns:
        dict with keys: original, corrected, model, run_index, system
    """
    import openai  # reuses the openai client, no extra library needed

    client = openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    prompt = _build_prompt(sentence)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content
            return {
                "system": "deepseek",
                "model": DEEPSEEK_MODEL,
                "run_index": run_index,
                "original": sentence,
                "corrected": _extract_correction(raw),
            }
        except openai.RateLimitError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
        except openai.APIError as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


# -----------------------------------------------------------------------
# Quick smoke test — run directly: python llm_wrappers.py
# -----------------------------------------------------------------------

if __name__ == "__main__":
    test_sentence = "She go to school yesterday."

    print(f"Prompt template:\n  {CORRECTION_PROMPT}\n")
    print(f"Test sentence: {test_sentence}\n")

    for fn, name in [(chatgpt_correct, "ChatGPT"),
                     (claude_correct,  "Claude"),
                     (deepseek_correct,"DeepSeek")]:
        # Check API key is set before attempting the call
        key_var = {"ChatGPT": "OPENAI_API_KEY",
                   "Claude": "ANTHROPIC_API_KEY",
                   "DeepSeek": "DEEPSEEK_API_KEY"}[name]
        if not os.environ.get(key_var):
            print(f"{name}: skipped — {key_var} not set in environment")
            continue
        try:
            result = fn(test_sentence, run_index=1)
            print(f"{name}:")
            print(f"  Original : {result['original']}")
            print(f"  Corrected: {result['corrected']}")
            print(f"  Model    : {result['model']}")
        except Exception as e:
            print(f"{name}: ERROR — {e}")
        print()
