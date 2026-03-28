import json
from groq import Groq
from shared_config import GROQ_KEY, ExecutionOutput, FinalOutput, METRICS_LOG

client = Groq(api_key=GROQ_KEY)

COMBINE_SYSTEM = """
You are a response synthesizer. Combine multiple AI responses into one
clear, coherent, well-structured final answer. Do not mention which model
produced which part. Write naturally as one assistant answering the whole question.
"""

def build_combine_prompt(original_prompt: str, results: list) -> str:
    parts  = [f"[Part {i}]\n{r['response']}" for i, r in enumerate(results, 1)]
    joined = "\n\n".join(parts)
    return f"Original question:\n{original_prompt}\n\nResponses:\n{joined}"

def compute_metrics(results: list) -> dict:
    latencies      = [r["latency_ms"]    for r in results]
    costs          = [r["cost_usd"]      for r in results]
    quality_scores = [r["quality_score"] for r in results]
    tokens         = [r["tokens_used"]   for r in results]
    models_used    = list({r["model_used"] for r in results})

    segment_breakdown = [{
        "segment":       r["segment_text"][:60] + ("..." if len(r["segment_text"]) > 60 else ""),
        "model_key":     r["model_used"],
        "model_name":    r["model_name"],
        "params_used":   r["params_used"],
        "latency_ms":    r["latency_ms"],
        "tokens_used":   r["tokens_used"],
        "cost_usd":      r["cost_usd"],
        "quality_score": r["quality_score"],
        "fallback_used": r.get("fallback_used", False),
    } for r in results]

    return {
        "total_segments": len(results),
        "models_used":    models_used,
        "latency": {
            "avg_ms":   round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "max_ms":   round(max(latencies), 1) if latencies else 0,
            "min_ms":   round(min(latencies), 1) if latencies else 0,
            "total_ms": round(sum(latencies), 1),
        },
        "cost": {
            "total_usd":   round(sum(costs), 6),
            "total_inr":   round(sum(costs) * 83.5, 4),
            "per_segment": [round(c, 6) for c in costs],
        },
        "quality": {
            "avg_score": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0,
            "max_score": max(quality_scores) if quality_scores else 0,
            "min_score": min(quality_scores) if quality_scores else 0,
        },
        "tokens": {
            "total":       sum(tokens),
            "per_segment": tokens,
        },
        "segment_breakdown": segment_breakdown,
    }

def save_metrics(metrics: dict):
    with open(METRICS_LOG, "w") as f:
        json.dump(metrics, f, indent=2)

def run_combiner(original_prompt: str, execution_output: ExecutionOutput) -> FinalOutput:
    results = execution_output["results"]

    if len(results) == 1:
        combined = results[0]["response"]
    else:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": COMBINE_SYSTEM},
                {"role": "user",   "content": build_combine_prompt(original_prompt, results)},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        combined = response.choices[0].message.content.strip()

    metrics = compute_metrics(results)
    save_metrics(metrics)
    return {
        "original_prompt":   original_prompt,
        "combined_response": combined,
        "metrics":           metrics,
    }

if __name__ == "__main__":
    mock = {
        "results": [{
            "segment_text": "Explain gravity",
            "model_used":   "simple",
            "model_name":   "llama-3.1-8b-instant",
            "params_used":  {"temperature": 0.5, "max_tokens": 512,
                             "top_p": 0.95, "note": "test"},
            "response":     "Gravity is the force attracting objects with mass.",
            "latency_ms":   340.0,
            "tokens_used":  45,
            "cost_usd":     0.000002,
            "quality_score": 6.5,
            "fallback_used": False,
        }]
    }
    print(json.dumps(run_combiner("Explain gravity", mock), indent=2))