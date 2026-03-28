"""
quality_scorer.py  ← NEW FILE
Dynamically calculates response quality scores using:
  - Heuristics only   → for "simple" complexity segments
  - Heuristics + LLM judge → for "medium" and "complex" segments

Final score formula:
  simple   → heuristic_score  (out of 10)
  medium/complex → (heuristic_score × 0.4) + (llm_judge_score × 0.6)
"""

import re
import json
from groq import Groq
from shared_config import GROQ_KEY

client = Groq(api_key=GROQ_KEY)

# ── LLM Judge ─────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """
You are a strict response quality evaluator.
Given a user prompt and an AI response, score the response from 0.0 to 10.0.

Return ONLY valid JSON with this exact structure:
{
  "score": <float between 0.0 and 10.0>,
  "reason": "<one concise sentence explaining the score>"
}

Scoring criteria:
- 9-10 : Complete, accurate, well-structured, addresses all parts of the prompt
- 7-8  : Mostly complete, minor gaps or slight inaccuracies
- 5-6  : Partially addresses the prompt, missing key parts
- 3-4  : Weak response, significant gaps or errors
- 0-2  : Irrelevant, wrong, or failed response

Be strict. Do not round up. Return ONLY the JSON — no markdown, no explanation.
"""

def llm_judge_score(prompt: str, response: str) -> tuple[float, str]:
    """
    Uses llama-3.1-8b-instant (fast + cheap) to score the response.
    Returns (score, reason).
    """
    try:
        result = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": (
                    f"PROMPT:\n{prompt}\n\n"
                    f"RESPONSE:\n{response}"
                )},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        raw = result.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())
        score  = float(parsed.get("score", 5.0))
        reason = parsed.get("reason", "No reason provided")
        return round(min(max(score, 0.0), 10.0), 2), reason

    except Exception as e:
        # Fallback if judge fails — don't crash the pipeline
        return 5.0, f"Judge failed: {str(e)[:60]}"


# ── Heuristics ────────────────────────────────────────────────────────────────

def score_length_utilization(response: str, max_tokens: int) -> float:
    """
    Checks if the response used the token budget reasonably.
    Estimated response tokens = len(response) / 4

    Scoring:
      < 15% of budget used → 3.0  (too short, likely incomplete)
      15–30%              → 5.0
      30–70%              → 8.5  (sweet spot)
      70–90%              → 9.5  (great utilization)
      > 90%               → 7.0  (possibly cut off)
    """
    if max_tokens <= 0:
        return 5.0
    estimated_tokens = len(response) / 4
    utilization      = estimated_tokens / max_tokens

    if utilization < 0.15:  return 3.0
    if utilization < 0.30:  return 5.0
    if utilization < 0.70:  return 8.5
    if utilization < 0.90:  return 9.5
    return 7.0


def score_keyword_coverage(prompt: str, response: str) -> float:
    """
    Checks what % of meaningful prompt words appear in the response.
    Filters out stopwords to focus on content words.
    Score = coverage% mapped to 0–10.
    """
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "if", "as", "it", "its", "this", "that",
        "me", "my", "you", "your", "we", "our", "they", "their",
        "what", "how", "why", "when", "where", "which", "who",
        "i", "he", "she", "him", "her", "us", "them",
    }

    prompt_words   = set(re.findall(r'\b[a-zA-Z]{3,}\b', prompt.lower()))
    response_lower = response.lower()
    content_words  = prompt_words - stopwords

    if not content_words:
        return 7.0  # Nothing meaningful to check

    covered = sum(1 for w in content_words if w in response_lower)
    coverage = covered / len(content_words)

    # Map coverage to score
    if coverage >= 0.85:  return 10.0
    if coverage >= 0.70:  return 8.5
    if coverage >= 0.55:  return 7.0
    if coverage >= 0.40:  return 5.5
    if coverage >= 0.25:  return 4.0
    return 2.5


def score_structure_signals(response: str, segment_type: str) -> float:
    """
    Checks for structural quality signals appropriate to the segment type.

    coding   → expects code blocks (``` or indented code)
    math     → expects numbered steps or equations
    creative → expects paragraphs, varied sentence length
    general  → expects coherent prose or lists
    """
    score = 5.0  # baseline

    has_code_block   = bool(re.search(r'```[\s\S]*?```', response))
    has_numbered     = bool(re.search(r'^\s*\d+[\.\)]\s', response, re.MULTILINE))
    has_bullets      = bool(re.search(r'^\s*[-•*]\s', response, re.MULTILINE))
    has_headings     = bool(re.search(r'^#{1,3}\s|\*\*[^*]+\*\*', response, re.MULTILINE))
    has_equations    = bool(re.search(r'[\=\+\-\×\÷\^√]|\\frac|\\sum', response))
    paragraph_count  = len([p for p in response.split('\n\n') if len(p.strip()) > 40])
    sentence_count   = len(re.findall(r'[.!?]+', response))
    avg_word_len     = (
        sum(len(w) for w in response.split()) / max(len(response.split()), 1)
    )

    if segment_type == "coding":
        if has_code_block:   score += 4.0
        if has_numbered:     score += 0.5
        if sentence_count > 2: score += 0.5  # has explanation too

    elif segment_type == "math":
        if has_numbered:     score += 3.0
        if has_equations:    score += 2.0
        if has_headings:     score += 0.5

    elif segment_type == "creative":
        if paragraph_count >= 2: score += 2.0
        if paragraph_count >= 4: score += 1.0
        if avg_word_len > 4.5:   score += 1.0  # richer vocabulary
        if sentence_count > 5:   score += 1.0

    elif segment_type == "general":
        if has_bullets or has_numbered: score += 2.0
        if paragraph_count >= 2:        score += 1.5
        if sentence_count > 3:          score += 1.0

    return round(min(score, 10.0), 2)


def score_latency_penalty(latency_ms: float) -> float:
    """
    Returns a penalty deduction based on response latency.
    Fast responses get 0 penalty; very slow ones lose up to 2 points.

      < 1500ms  → 0.0 penalty
      1500–3000 → 0.5 penalty
      3000–5000 → 1.0 penalty
      > 5000ms  → 2.0 penalty
    """
    if latency_ms < 1500:  return 0.0
    if latency_ms < 3000:  return 0.5
    if latency_ms < 5000:  return 1.0
    return 2.0


def compute_heuristic_score(
    prompt:       str,
    response:     str,
    segment_type: str,
    max_tokens:   int,
    latency_ms:   float,
) -> dict:
    """
    Combines all 4 heuristics into a weighted score out of 10.

    Weights:
      keyword_coverage    → 35%
      structure_signals   → 30%
      length_utilization  → 25%
      latency_penalty     → deducted at end (up to -2pts)
    """
    length_score    = score_length_utilization(response, max_tokens)
    keyword_score   = score_keyword_coverage(prompt, response)
    structure_score = score_structure_signals(response, segment_type)
    latency_penalty = score_latency_penalty(latency_ms)

    raw_score = (
        keyword_score   * 0.35
        + structure_score * 0.30
        + length_score    * 0.25
        # latency is a flat deduction, not a weighted factor
    )

    final = round(max(0.0, min(raw_score - latency_penalty, 10.0)), 2)

    return {
        "heuristic_score":    final,
        "breakdown": {
            "length_score":    length_score,
            "keyword_score":   keyword_score,
            "structure_score": structure_score,
            "latency_penalty": -latency_penalty,
        }
    }


# ── Main Entry Point ──────────────────────────────────────────────────────────

def calculate_quality_score(
    prompt:       str,
    response:     str,
    segment_type: str,
    complexity:   str,
    max_tokens:   int,
    latency_ms:   float,
) -> dict:
    """
    Main function called from person4_execution.py

    Returns a dict with:
      quality_score     → final score (0.0 – 10.0)
      method            → "heuristics_only" or "heuristics+llm_judge"
      heuristic_score   → raw heuristic score
      llm_judge_score   → LLM score (None if not used)
      llm_judge_reason  → explanation from LLM judge (None if not used)
      breakdown         → per-heuristic scores
    """

    heuristic_result = compute_heuristic_score(
        prompt, response, segment_type, max_tokens, latency_ms
    )
    h_score   = heuristic_result["heuristic_score"]
    breakdown = heuristic_result["breakdown"]

    # Simple segments → heuristics only (no extra API call)
    if complexity == "simple":
        return {
            "quality_score":    h_score,
            "method":           "heuristics_only",
            "heuristic_score":  h_score,
            "llm_judge_score":  None,
            "llm_judge_reason": None,
            "breakdown":        breakdown,
        }

    # Medium / Complex → heuristics + LLM judge
    j_score, j_reason = llm_judge_score(prompt, response)
    final_score       = round((h_score * 0.4) + (j_score * 0.6), 2)

    return {
        "quality_score":    final_score,
        "method":           "heuristics+llm_judge",
        "heuristic_score":  h_score,
        "llm_judge_score":  j_score,
        "llm_judge_reason": j_reason,
        "breakdown":        breakdown,
    }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "prompt":       "Write a Python function to calculate factorial",
            "response":     "```python\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n```\nThis recursive function handles the base case of 0 and calls itself for larger values.",
            "segment_type": "coding",
            "complexity":   "medium",
            "max_tokens":   1024,
            "latency_ms":   800.0,
        },
        {
            "prompt":       "What is photosynthesis?",
            "response":     "Photosynthesis is how plants make food from sunlight.",
            "segment_type": "general",
            "complexity":   "simple",
            "max_tokens":   256,
            "latency_ms":   300.0,
        },
        {
            "prompt":       "Prove the Pythagorean theorem step by step",
            "response":     "Sure.",
            "segment_type": "math",
            "complexity":   "complex",
            "max_tokens":   2048,
            "latency_ms":   6000.0,
        },
    ]

    for tc in test_cases:
        print(f"\nPrompt   : {tc['prompt'][:60]}")
        print(f"Type     : {tc['segment_type']} | Complexity: {tc['complexity']}")
        result = calculate_quality_score(**tc)
        print(f"Method   : {result['method']}")
        print(f"Score    : {result['quality_score']} / 10")
        print(f"Heuristic: {result['heuristic_score']}")
        if result["llm_judge_score"] is not None:
            print(f"LLM Judge: {result['llm_judge_score']} — {result['llm_judge_reason']}")
        print(f"Breakdown: {result['breakdown']}")