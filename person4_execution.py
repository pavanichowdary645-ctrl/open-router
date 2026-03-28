

import time
import concurrent.futures
from groq import Groq
from shared_config import (
    GROQ_KEY, MODELS,
    MODEL_COST_PER_1K_TOKENS,
    DecisionOutput, ExecutionOutput, ExecutionItem,
)
from quality_scorer import calculate_quality_score   # ← NEW

client = Groq(api_key=GROQ_KEY)

# Fallback chain per model key
FALLBACK_CHAIN = {
    "math":     ["openai/gpt-oss-120b",    "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "coding":   ["qwen/qwen3-32b",          "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "creative": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "simple":   ["llama-3.1-8b-instant",    "llama-3.3-70b-versatile"],
    "general":  ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
}

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def calculate_cost(model_name: str, input_text: str, output_text: str) -> float:
    total_tokens = estimate_tokens(input_text) + estimate_tokens(output_text)
    rate = MODEL_COST_PER_1K_TOKENS.get(model_name, 0.0005)
    return round((total_tokens / 1000) * rate, 6)

def call_with_fallback(text: str, model_key: str, params: dict) -> dict:
    chain      = FALLBACK_CHAIN.get(model_key, ["llama-3.3-70b-versatile"])
    last_error = None

    for model_name in chain:
        for retry in range(1, 3):
            try:
                t0 = time.time()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": text}],
                    temperature=params.get("temperature", 0.5),
                    max_tokens=params.get("max_tokens", 512),
                    top_p=params.get("top_p", 0.95),
                )
                latency = round((time.time() - t0) * 1000, 1)
                return {
                    "response":      response.choices[0].message.content,
                    "model_name":    model_name,
                    "latency_ms":    latency,
                    "fallback_used": model_name != chain[0],
                }
            except Exception as e:
                last_error = str(e)
                if "decommissioned" in last_error or "not found" in last_error:
                    print(f"  [fallback] {model_name} decommissioned → next")
                    break
                if "429" in last_error:
                    print(f"  [fallback] Rate limit on {model_name} → retry {retry}")
                    time.sleep(2 * retry)
                    continue
                print(f"  [fallback] {model_name} failed → next")
                break

    return {
        "response":      f"ERROR: All fallbacks exhausted. {last_error}",
        "model_name":    "none",
        "latency_ms":    0.0,
        "fallback_used": True,
    }


def execute_one(item: dict, segment_type: str, complexity: str) -> ExecutionItem:
    """
    Executes one segment and calculates its quality score dynamically.

    Args:
        item:         decision item (text, model, params)
        segment_type: from analyzer output (math/coding/creative/general)
        complexity:   from analyzer output (simple/medium/complex)
    """
    model_key = item["model"]
    params    = item["params"]
    result    = call_with_fallback(item["text"], model_key, params)

    actual_model = result["model_name"]
    response     = result["response"]
    latency      = result["latency_ms"]
    tokens_used  = estimate_tokens(item["text"]) + estimate_tokens(response)
    cost_usd     = calculate_cost(actual_model, item["text"], response)

    # ── Dynamic quality scoring ──────────────────────────────────────────────
    quality_result = calculate_quality_score(
        prompt=       item["text"],
        response=     response,
        segment_type= segment_type,
        complexity=   complexity,
        max_tokens=   params.get("max_tokens", 512),
        latency_ms=   latency,
    )
    quality_score  = quality_result["quality_score"]
    quality_detail = {
        "method":           quality_result["method"],
        "heuristic_score":  quality_result["heuristic_score"],
        "llm_judge_score":  quality_result["llm_judge_score"],
        "llm_judge_reason": quality_result["llm_judge_reason"],
        "breakdown":        quality_result["breakdown"],
    }
    # ────────────────────────────────────────────────────────────────────────

    return {
        "segment_text":   item["text"],
        "segment_type":   segment_type,       # ← added for traceability
        "complexity":     complexity,          # ← added for traceability
        "model_used":     model_key,
        "model_name":     actual_model,
        "params_used": {
            "temperature": params.get("temperature"),
            "max_tokens":  params.get("max_tokens"),
            "top_p":       params.get("top_p"),
            "note":        params.get("note", ""),
        },
        "response":       response,
        "latency_ms":     latency,
        "tokens_used":    tokens_used,
        "cost_usd":       cost_usd,
        "quality_score":  quality_score,      # ← now dynamic
        "quality_detail": quality_detail,     # ← new: full scoring breakdown
        "fallback_used":  result.get("fallback_used", False),
    }


def run_execution(decision_output: DecisionOutput, analyzer_output: dict) -> ExecutionOutput:
    """
    Runs all segments in parallel.

    Args:
        decision_output: from person3_decision.run_decision()
        analyzer_output: from person2_analyzer.run_analyzer()
                         needed to pass segment_type + complexity to scorer
    """
    items      = decision_output["items"]
    complexity = analyzer_output["complexity"]
    segments   = analyzer_output["segments"]

    # Build a lookup: segment text → segment type
    type_lookup = {seg["text"]: seg["type"] for seg in segments}

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                execute_one,
                item,
                type_lookup.get(item["text"], "general"),  # segment_type
                complexity,
            ): item
            for item in items
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return {"results": results}


if __name__ == "__main__":
    import json

    mock_analyzer = {
        "complexity": "medium",
        "segments": [
            {"text": "What is the capital of France?", "type": "general"},
        ]
    }
    mock_decision = {
        "items": [{
            "text":   "What is the capital of France?",
            "model":  "simple",
            "params": {"temperature": 0.5, "max_tokens": 256, "top_p": 0.95, "note": "test"}
        }]
    }

    out = run_execution(mock_decision, mock_analyzer)
    print(json.dumps(out, indent=2))