import json
import os
from datetime import datetime
from shared_config import ROUTING_MEMORY

def load_memory() -> dict:
    if not os.path.exists(ROUTING_MEMORY):
        return {}
    with open(ROUTING_MEMORY, "r") as f:
        return json.load(f)

def save_memory(memory: dict):
    with open(ROUTING_MEMORY, "w") as f:
        json.dump(memory, f, indent=2)

def record_routing_decision(
    segment_type:  str,
    complexity:    str,
    model_used:    str,
    latency_ms:    float,
    cost_usd:      float,
    quality_score: float,
    tokens_used:   int,
):
    memory = load_memory()
    key    = f"{segment_type}_{complexity}"

    if key not in memory:
        memory[key] = {}

    if model_used not in memory[key]:
        memory[key][model_used] = {
            "total_calls":     0,
            "avg_latency_ms":  0.0,
            "avg_cost_usd":    0.0,
            "avg_quality":     0.0,
            "avg_tokens":      0,
            "composite_score": 0.0,
            "last_used":       "",
        }

    e = memory[key][model_used]
    n = e["total_calls"]

    # Rolling average
    e["avg_latency_ms"] = round((e["avg_latency_ms"] * n + latency_ms)    / (n + 1), 2)
    e["avg_cost_usd"]   = round((e["avg_cost_usd"]   * n + cost_usd)      / (n + 1), 6)
    e["avg_quality"]    = round((e["avg_quality"]     * n + quality_score) / (n + 1), 2)
    e["avg_tokens"]     = int((e["avg_tokens"]        * n + tokens_used)   / (n + 1))
    e["total_calls"]    = n + 1
    e["last_used"]      = datetime.now().isoformat()

    # Composite score: quality (50%) + cost savings (30%) + speed (20%)
    latency_norm = min(e["avg_latency_ms"] / 5000.0, 1.0)
    cost_norm    = min(e["avg_cost_usd"]   / 0.01,   1.0)
    quality_norm = e["avg_quality"] / 10.0

    e["composite_score"] = round(
        quality_norm * 0.5
        + (1 - cost_norm)    * 0.3
        + (1 - latency_norm) * 0.2,
        4
    )

    save_memory(memory)
    return e["composite_score"]

def get_best_model_from_memory(segment_type: str, complexity: str):
    memory = load_memory()
    key    = f"{segment_type}_{complexity}"

    if key not in memory or not memory[key]:
        return None

    # Only trust memory after 1 data points
    qualified = {
        m: s for m, s in memory[key].items()
        if s["total_calls"] >= 1
    }
    if not qualified:
        return None

    return max(qualified.items(), key=lambda x: x[1]["composite_score"])[0]

def get_learning_status(segment_type: str, complexity: str) -> dict:
    memory = load_memory()
    key    = f"{segment_type}_{complexity}"

    if key not in memory or not memory[key]:
        return {
            "status":       "learning",
            "calls_needed": 1,
            "calls_done":   0,
            "progress":     "0/1 runs collected — no data yet",
            "best_model":   None,
            "score":        0,
        }

    max_calls = max(v["total_calls"] for v in memory[key].values())

    if max_calls < 1:
        return {
            "status":       "learning",
            "calls_needed": 1,
            "calls_done":   max_calls,
            "progress":     f"{max_calls}/1 runs collected",
            "best_model":   None,
            "score":        0,
        }

    best = get_best_model_from_memory(segment_type, complexity)
    best_score = memory[key][best]["composite_score"] if best else 0

    return {
        "status":      "learned",
        "best_model":  best,
        "score":       best_score,
        "total_calls": max_calls,
        "progress":    f"Routing optimized after {max_calls} run(s)",
        "calls_done":  max_calls,
        "calls_needed": 1,
    }

def print_learning_report():
    memory = load_memory()
    if not memory:
        print("\n  [LEARNING] No routing memory yet — run more prompts to train the router.")
        return

    print("\n  ┌──────────────────────────────────────────────────────────────────┐")
    print(  "  │                  SELF-LEARNING ROUTER REPORT                     │")
    print(  "  ├──────────────────────────────────────────────────────────────────┤")

    for key, models in memory.items():
        seg_type, compl = key.split("_", 1)
        status          = get_learning_status(seg_type, compl)

        print(f"\n  Task Type : {seg_type.upper()} | Complexity: {compl}")
        print(f"  Status    : {status['status'].upper()} — {status['progress']}")

        if status["status"] == "learned":
            print(f"  Best Model: {status['best_model']} "
                  f"(composite score: {status['score']})")

        print(f"  {'Model':<35} {'Calls':<6} {'Score':<8} "
              f"{'Quality':<10} {'Latency':<12} {'Cost'}")
        print("  " + "─" * 82)

        for model, stats in sorted(
            models.items(),
            key=lambda x: x[1]["composite_score"],
            reverse=True
        ):
            best_tag = " ← BEST" if (
                status["status"] == "learned"
                and model == status.get("best_model")
            ) else ""
            print(f"  {model:<35} "
                  f"{stats['total_calls']:<6} "
                  f"{stats['composite_score']:<8} "
                  f"{stats['avg_quality']}/10     "
                  f"{stats['avg_latency_ms']}ms      "
                  f"${stats['avg_cost_usd']:.6f}"
                  f"{best_tag}")

    print("\n  └──────────────────────────────────────────────────────────────────┘")

if __name__ == "__main__":
    # Simulate 4 runs to trigger learning
    for i in range(4):
        record_routing_decision("math", "complex", "openai/gpt-oss-120b",
                                1200, 0.00045, 9.5, 380)
        record_routing_decision("creative", "medium", "llama-3.3-70b-versatile",
                                900, 0.00032, 8.5, 280)

    print_learning_report()
    print("\nBest for math/complex  :", get_best_model_from_memory("math", "complex"))
    print("Best for creative/medium:", get_best_model_from_memory("creative", "medium"))