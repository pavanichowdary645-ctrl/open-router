import json, os, time
from shared_config import (
    OUTPUT_DIR, ANALYZER_OUT, DECISION_OUT, EXECUTION_OUT, FINAL_OUT
)
from person2_analyzer  import run_analyzer
from person3_decision  import run_decision
from person4_execution import run_execution
from person5_combiner  import run_combiner
from person6_feedback  import (
    record_routing_decision,
    get_best_model_from_memory,
    get_learning_status,
    print_learning_report,
)
from person7_rewards import update_rewards, print_rewards

def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [saved] {path}")

def print_segment_table(results: list):
    print("\n  ┌──────────────┬──────────────┬────────────┬"
          "──────────┬──────────┬─────────┬──────────┐")
    print(  "  │ Model key    │ Temp / Tokens│  Latency   │"
            "  Tokens  │  Cost    │ Quality │ Fallback │")
    print(  "  ├──────────────┼──────────────┼────────────┼"
            "──────────┼──────────┼─────────┼──────────┤")
    for r in results:
        p = r["params_used"]
        fb = "YES" if r.get("fallback_used") else "no"
        print(f"  │ {r['model_used']:<12} │ "
              f"{str(p.get('temperature','?')):<5}/"
              f"{str(p.get('max_tokens','?')):<6} │ "
              f"{r['latency_ms']:>8.1f}ms │ "
              f"{r['tokens_used']:>8} │ "
              f"${r['cost_usd']:.5f} │ "
              f"{r['quality_score']:>5.1f}/10 │ "
              f"{fb:<8} │")
    print(  "  └──────────────┴──────────────┴────────────┴"
            "──────────┴──────────┴─────────┴──────────┘")

def print_metrics_summary(metrics: dict):
    print("\n  ┌─────────────────────────────────────────┐")
    print(  "  │           PIPELINE METRICS              │")
    print(  "  ├─────────────────────────────────────────┤")
    print(f"  │  Total segments  : {metrics['total_segments']:<21}│")
    print(f"  │  Models used     : {', '.join(metrics['models_used']):<21}│")
    print(  "  ├─────────────────────────────────────────┤")
    print(f"  │  Avg latency     : {metrics['latency']['avg_ms']:>8.1f} ms         │")
    print(f"  │  Min latency     : {metrics['latency']['min_ms']:>8.1f} ms         │")
    print(f"  │  Max latency     : {metrics['latency']['max_ms']:>8.1f} ms         │")
    print(f"  │  Total latency   : {metrics['latency']['total_ms']:>8.1f} ms         │")
    print(  "  ├─────────────────────────────────────────┤")
    print(f"  │  Total cost (USD): ${metrics['cost']['total_usd']:.6f}             │")
    print(f"  │  Total cost (INR): ₹{metrics['cost']['total_inr']:.4f}               │")
    print(  "  ├─────────────────────────────────────────┤")
    print(f"  │  Avg quality     : {metrics['quality']['avg_score']:>8.2f} / 10        │")
    print(f"  │  Best quality    : {metrics['quality']['max_score']:>8.2f} / 10        │")
    print(  "  ├─────────────────────────────────────────┤")
    print(f"  │  Total tokens    : {metrics['tokens']['total']:<21}│")
    print(  "  └─────────────────────────────────────────┘")

def run_pipeline(user_prompt: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 50)
    print("       MULTI-MODEL INTELLIGENT PIPELINE")
    print("=" * 50)
    print(f"  Prompt: {user_prompt}\n")

    t_total = time.time()

    # ── Step 1 — Analyze ──────────────────────────────
    print("[Step 1/4] Analyzing prompt...")
    analyzer_result = run_analyzer(user_prompt)
    save_json(ANALYZER_OUT, analyzer_result)
    print(f"  Complexity : {analyzer_result['complexity']}")
    print(f"  Segments   : {len(analyzer_result['segments'])}")
    for i, seg in enumerate(analyzer_result["segments"], 1):
        print(f"    Segment {i}: [{seg['type'].upper()}] {seg['text'][:65]}")
    print()

    # ── Step 2 — Decide + Learning Override ───────────
    print("[Step 2/4] Running decision engine + self-learning check...")
    decision_result = run_decision(analyzer_result)

    print("\n  SELF-LEARNING STATUS:")
    print("  " + "─" * 60)
    for item in decision_result["items"]:
        seg = next(
            s for s in analyzer_result["segments"]
            if s["text"] == item["text"]
        )
        seg_type   = seg["type"]
        complexity = analyzer_result["complexity"]
        status     = get_learning_status(seg_type, complexity)
        best_model = get_best_model_from_memory(seg_type, complexity)

        print(f"  Segment type : {seg_type.upper()}")
        print(f"  Learn status : {status['status'].upper()} — {status['progress']}")

        if best_model and best_model != item["model"]:
            print(f"  Override     : {item['model']} → {best_model} "
                  f"(learned best model, score: {status.get('score', 'N/A')})")
            item["model"] = best_model
        elif status["status"] == "learned":
            print(f"  Decision     : keeping {item['model']} "
                  f"(already optimal, score: {status.get('score', 'N/A')})")
        else:
            calls_done   = status.get("calls_done", 0)
            calls_needed = status.get("calls_needed", 3)
            remaining    = calls_needed - calls_done
            print(f"  Decision     : using default {item['model']} "
                  f"({remaining} more run(s) needed to learn)")

        print("  " + "─" * 60)

    save_json(DECISION_OUT, decision_result)

    print("\n  DYNAMIC PARAMETERS:")
    print("  " + "─" * 60)
    for item in decision_result["items"]:
        p = item["params"]
        print(f"  Segment  : {item['text'][:65]}")
        print(f"  Model    : {item['model']}")
        print(f"  Difficulty Score     : {p['difficulty_score']}/10")
        print(f"  Reason   : {p['note']}")
        print(f"  ► temperature        = {p['temperature']}")
        print(f"  ► max_tokens         = {p['max_tokens']}")
        print(f"  ► top_p              = {p['top_p']}")
        print(f"  ► frequency_penalty  = {p['frequency_penalty']}")
        print("  " + "─" * 60)
    print()

    # ── Step 3 — Execute ──────────────────────────────
    print("[Step 3/4] Executing across models (parallel)...")
    execution_result = run_execution(decision_result,analyzer_result)
    save_json(EXECUTION_OUT, execution_result)
    print_segment_table(execution_result["results"])
    print()

    # ── Step 4 — Combine ──────────────────────────────
    print("[Step 4/4] Combining responses...")
    final_result = run_combiner(user_prompt, execution_result)
    save_json(FINAL_OUT, final_result)

    total_ms = (time.time() - t_total) * 1000

    # ── Record to self-learning memory ────────────────
    for r in execution_result["results"]:
        seg_type = next(
            (s["type"] for s in analyzer_result["segments"]
             if s["text"] == r["segment_text"]),
            "general"
        )
        record_routing_decision(
            segment_type=  seg_type,
            complexity=    analyzer_result["complexity"],
            model_used=    r["model_name"],
            latency_ms=    r["latency_ms"],
            cost_usd=      r["cost_usd"],
            quality_score= r["quality_score"],
            tokens_used=   r["tokens_used"],
        )

    # ── Rewards ───────────────────────────────────────
    reward_data = update_rewards(execution_result["results"], user_prompt)

    # ── Print all outputs ─────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  DONE in {total_ms:.0f}ms")
    print(f"{'=' * 50}")

    print_metrics_summary(final_result["metrics"])
    print_rewards(reward_data)
    print_learning_report()

    print("\n--- FINAL RESPONSE ---\n")
    print(final_result["combined_response"])

    print("\n--- DYNAMIC PARAMETERS USED PER SEGMENT ---")
    for r in execution_result["results"]:
        print(f"\n  Segment  : {r['segment_text'][:65]}")
        print(f"  Model    : {r['model_name']} (key: {r['model_used']})")
        print(f"  Params   : temp={r['params_used']['temperature']}  "
              f"tokens={r['params_used']['max_tokens']}  "
              f"top_p={r['params_used']['top_p']}")
        print(f"  Note     : {r['params_used'].get('note', '')}")

    return {
        "final": final_result,
        "analyzer": analyzer_result,
        "execution": execution_result,
        "reward_data": reward_data,
    }

if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    run_pipeline(prompt)