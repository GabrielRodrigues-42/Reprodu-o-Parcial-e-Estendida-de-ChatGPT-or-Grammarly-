import requests

def languagetool_correct(sentence, language="en-US", endpoint="http://localhost:8010/v2/check"):
    """
    Sends a sentence to a local LanguageTool server and returns a corrected
    version, applying the first suggested replacement for each detected match.

    Returns a dict with the corrected sentence plus the raw matches, so you
    keep the detailed info (rule IDs, categories) for later analysis even
    though you're collapsing it to a single string here.
    """
    response = requests.post(
        endpoint,
        data={"text": sentence, "language": language}
    )
    response.raise_for_status()
    result = response.json()

    matches = result.get("matches", [])

    # Apply right-to-left so earlier offsets aren't invalidated by edits
    # made later in the sentence.
    corrected = sentence
    for match in sorted(matches, key=lambda m: m["offset"], reverse=True):
        replacements = match.get("replacements", [])
        if not replacements:
            continue  # some matches (style flags etc.) have no suggestion
        chosen = replacements[0]["value"]  # first suggestion = default choice
        start = match["offset"]
        end = start + match["length"]
        corrected = corrected[:start] + chosen + corrected[end:]

    return {
        "original": sentence,
        "corrected": corrected,
        "num_matches": len(matches),
        "matches": matches,  # keep raw data for later error-type analysis
    }


if __name__ == "__main__":
    test_sentences = [
        "She go to school yesterday.",
        "He don't like apples and she dont either.",
    ]
    for s in test_sentences:
        result = languagetool_correct(s)
        print(f"Original : {result['original']}")
        print(f"Corrected: {result['corrected']}")
        print(f"Matches  : {result['num_matches']}")
        print()