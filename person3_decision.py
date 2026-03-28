from shared_config import AnalyzerOutput, DecisionOutput, DecisionItem

TYPE_TO_MODEL = {
    "math":     "math",
    "coding":   "coding",
    "creative": "creative",
    "general":  "simple",
}

def select_model(segment_type: str, complexity: str) -> str:
    base = TYPE_TO_MODEL.get(segment_type, "simple")
    if segment_type == "general" and complexity == "complex":
        base = "math"
    return base

def count_words(text: str) -> int:
    return len(text.strip().split())

def has_multipart(text: str) -> bool:
    keywords = ["and", "also", "additionally", "then",
                "furthermore", "explain", "describe", "compare"]
    text_lower = text.lower()
    return sum(1 for k in keywords if k in text_lower) >= 2

def detect_depth_keywords(text: str) -> int:
    depth_words = [
        "detailed", "in depth", "step by step", "thoroughly",
        "elaborate", "comprehensive", "advanced", "derive", "prove",
        "implement", "optimize", "compare", "analyze", "evaluate",
        "hard", "complex", "difficult", "intricate", "sophisticated"
    ]
    text_lower = text.lower()
    return sum(1 for w in depth_words if w in text_lower)

def compute_dynamic_params(
    segment_text: str,
    segment_type: str,
    complexity: str
) -> dict:
    word_count     = count_words(segment_text)
    question_count = segment_text.count("?")
    multipart      = has_multipart(segment_text)
    depth_score    = detect_depth_keywords(segment_text)

    complexity_weight = {"simple": 0, "medium": 1, "complex": 2}.get(complexity, 1)

    difficulty = min(10, (
        complexity_weight * 2.5
        + min(depth_score, 4) * 1.0
        + min(word_count / 20, 2.0)
        + question_count * 0.5
        + (1.0 if multipart else 0.0)
    ))

    base = {
        "creative": {"temp_min": 0.7,  "temp_max": 1.0,
                     "tokens_min": 512,  "tokens_max": 2048},
        "math":     {"temp_min": 0.0,  "temp_max": 0.4,
                     "tokens_min": 512,  "tokens_max": 2048},
        "coding":   {"temp_min": 0.1,  "temp_max": 0.5,
                     "tokens_min": 1024, "tokens_max": 4096},
        "general":  {"temp_min": 0.3,  "temp_max": 0.8,
                     "tokens_min": 256,  "tokens_max": 1024},
    }.get(segment_type,
          {"temp_min": 0.3, "temp_max": 0.8,
           "tokens_min": 256, "tokens_max": 1024})

    ratio       = difficulty / 10.0
    temperature = round(
        base["temp_min"] + ratio * (base["temp_max"] - base["temp_min"]), 2
    )
    max_tokens  = int(
        base["tokens_min"] + ratio * (base["tokens_max"] - base["tokens_min"])
    )
    top_p       = round(min(1.0, 0.85 + ratio * 0.15), 2)
    freq_pen    = round(0.1 + ratio * 0.4, 2) if segment_type == "creative" else 0.0

    reasons = []
    if complexity_weight >= 2:    reasons.append("complex prompt")
    if depth_score > 0:           reasons.append(f"{depth_score} depth keyword(s)")
    if multipart:                 reasons.append("multi-part request")
    if word_count > 20:           reasons.append(f"long prompt ({word_count} words)")
    if question_count > 1:        reasons.append(f"{question_count} questions")
    if not reasons:               reasons.append("straightforward prompt")

    return {
        "temperature":        temperature,
        "max_tokens":         max_tokens,
        "top_p":              top_p,
        "frequency_penalty":  freq_pen,
        "difficulty_score":   round(difficulty, 2),
        "word_count":         word_count,
        "depth_keywords_hit": depth_score,
        "multipart_detected": multipart,
        "note": f"difficulty={difficulty:.1f}/10 | {', '.join(reasons)}",
    }

def run_decision(analyzer_output: AnalyzerOutput) -> DecisionOutput:
    complexity = analyzer_output["complexity"]
    items: list[DecisionItem] = []
    for seg in analyzer_output["segments"]:
        model_key = select_model(seg["type"], complexity)
        params    = compute_dynamic_params(seg["text"], seg["type"], complexity)
        items.append({
            "text":   seg["text"],
            "model":  model_key,
            "params": params,
        })
    return {"items": items}

if __name__ == "__main__":
    import json
    mock = {
        "complexity": "complex",
        "segments": [
            {"text": "Prove Pythagorean theorem step by step", "type": "math"},
            {"text": "Write a detailed creative story about it", "type": "creative"},
            {"text": "Implement it in Python with edge cases",  "type": "coding"},
        ]
    }
    out = run_decision(mock)
    print(json.dumps(out, indent=2))