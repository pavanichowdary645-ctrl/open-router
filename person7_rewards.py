import json
import os
from datetime import datetime
from shared_config import CREDITS_FILE, EXPENSIVE_MODEL_COST_PER_1K

def load_credits() -> dict:
    if not os.path.exists(CREDITS_FILE):
        return {
            "total_credits":   0.0,
            "total_saved_usd": 0.0,
            "total_saved_inr": 0.0,
            "total_runs":      0,
            "lifetime_tokens": 0,
            "history":         [],
            "badges":          [],
        }
    with open(CREDITS_FILE, "r") as f:
        return json.load(f)

def save_credits(data: dict):
    with open(CREDITS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def calculate_reward(results: list) -> dict:
    total_actual_cost = sum(r["cost_usd"]    for r in results)
    total_tokens      = sum(r["tokens_used"] for r in results)
    baseline_cost     = (total_tokens / 1000) * EXPENSIVE_MODEL_COST_PER_1K
    saved_usd         = max(0.0, baseline_cost - total_actual_cost)
    credits_earned    = round(saved_usd * 1000, 4)
    efficiency_pct    = round(
        (saved_usd / baseline_cost * 100) if baseline_cost > 0 else 0, 1
    )

    # Per model breakdown
    model_breakdown = {}
    for r in results:
        m = r["model_used"]
        if m not in model_breakdown:
            model_breakdown[m] = {"cost": 0.0, "tokens": 0, "segments": 0}
        model_breakdown[m]["cost"]     += r["cost_usd"]
        model_breakdown[m]["tokens"]   += r["tokens_used"]
        model_breakdown[m]["segments"] += 1

    return {
        "baseline_cost_usd": round(baseline_cost,     6),
        "actual_cost_usd":   round(total_actual_cost, 6),
        "saved_usd":         round(saved_usd,         6),
        "saved_inr":         round(saved_usd * 83.5,  4),
        "credits_earned":    credits_earned,
        "efficiency_pct":    efficiency_pct,
        "total_tokens":      total_tokens,
        "model_breakdown":   model_breakdown,
    }

def assign_badges(data: dict, reward: dict) -> list:
    new_badges = []
    existing   = set(data["badges"])

    badge_rules = [
        ("🥉 First Save",         data["total_runs"] >= 1),
        ("🥈 Cost Saver",         data["total_saved_usd"] >= 0.0001),
        ("🥇 Smart Router",       data["total_saved_usd"] >= 0.001),
        ("🏆 Efficiency Master",  reward["efficiency_pct"] >= 60),
        ("💎 Credit Collector",   data["total_credits"] >= 1.0),
        ("⚡ Speed Optimizer",    reward["efficiency_pct"] >= 80),
        ("🔥 Power User",         data["total_runs"] >= 10),
        ("🌟 Token Saver",        data["lifetime_tokens"] >= 5000),
    ]

    for badge, condition in badge_rules:
        if condition and badge not in existing:
            new_badges.append(badge)
            data["badges"].append(badge)

    return new_badges

def update_rewards(results: list, prompt: str) -> dict:
    reward = calculate_reward(results)
    data   = load_credits()

    data["total_runs"]      += 1
    data["total_saved_usd"]  = round(data["total_saved_usd"] + reward["saved_usd"],      6)
    data["total_saved_inr"]  = round(data["total_saved_inr"] + reward["saved_inr"],      4)
    data["total_credits"]    = round(data["total_credits"]   + reward["credits_earned"], 4)
    data["lifetime_tokens"] += reward["total_tokens"]

    new_badges = assign_badges(data, reward)

    data["history"].append({
        "run":             data["total_runs"],
        "timestamp":       datetime.now().isoformat(),
        "prompt_preview":  prompt[:50],
        "credits_earned":  reward["credits_earned"],
        "saved_usd":       reward["saved_usd"],
        "efficiency_pct":  reward["efficiency_pct"],
        "models_used":     list(reward["model_breakdown"].keys()),
    })

    save_credits(data)

    return {
        "reward":          reward,
        "total_credits":   data["total_credits"],
        "total_saved_usd": data["total_saved_usd"],
        "total_saved_inr": data["total_saved_inr"],
        "total_runs":      data["total_runs"],
        "all_badges":      data["badges"],
        "new_badges":      new_badges,
        "history":         data["history"][-5:],
    }

def print_rewards(reward_data: dict):
    r = reward_data["reward"]

    print("\n  ┌────────────────────────────────────────────────────┐")
    print(  "  │              CREDIT & REWARD SYSTEM                │")
    print(  "  ├────────────────────────────────────────────────────┤")
    print(  "  │  THIS RUN                                          │")
    print(f"  │  Baseline cost (if expensive model used)           │")
    print(f"  │    ${r['baseline_cost_usd']:.6f}                              │")
    print(f"  │  Actual cost   (smart routing saved you)           │")
    print(f"  │    ${r['actual_cost_usd']:.6f}                              │")
    print(f"  │  Saved         : ${r['saved_usd']:.6f}  (₹{r['saved_inr']:.4f})   │")
    print(f"  │  Efficiency    : {r['efficiency_pct']}%                             │")
    print(f"  │  Credits earned: {r['credits_earned']}                           │")
    print(  "  ├────────────────────────────────────────────────────┤")
    print(  "  │  MODEL BREAKDOWN                                   │")
    for model, info in r["model_breakdown"].items():
        print(f"  │  {model:<20} → "
              f"{info['segments']} seg(s), "
              f"{info['tokens']} tokens, "
              f"${info['cost']:.6f}   │")
    print(  "  ├────────────────────────────────────────────────────┤")
    print(  "  │  LIFETIME TOTALS                                   │")
    print(f"  │  Total credits    : {reward_data['total_credits']:<31}│")
    print(f"  │  Total saved (USD): ${reward_data['total_saved_usd']:.6f}                   │")
    print(f"  │  Total saved (INR): ₹{reward_data['total_saved_inr']:.4f}                     │")
    print(f"  │  Total runs       : {reward_data['total_runs']:<31}│")
    print(  "  ├────────────────────────────────────────────────────┤")

    if reward_data["new_badges"]:
        print(  "  │  🎉 NEW BADGES UNLOCKED!                           │")
        for b in reward_data["new_badges"]:
            print(f"  │    {b:<47}│")
        print(  "  ├────────────────────────────────────────────────────┤")

    if reward_data["all_badges"]:
        print(  "  │  YOUR BADGES                                       │")
        for b in reward_data["all_badges"]:
            print(f"  │    {b:<47}│")

    print(  "  ├────────────────────────────────────────────────────┤")
    print(  "  │  LAST 5 RUNS                                       │")
    for h in reward_data["history"]:
        print(f"  │  Run #{h['run']:<3} | eff:{h['efficiency_pct']:>5}% | "
              f"credits:{h['credits_earned']:<7} | "
              f"saved:${h['saved_usd']:.6f}  │")
    print(  "  └────────────────────────────────────────────────────┘")

if __name__ == "__main__":
    mock_results = [
        {"cost_usd": 0.000002, "tokens_used": 50,
         "model_used": "simple",   "quality_score": 6.5},
        {"cost_usd": 0.000024, "tokens_used": 120,
         "model_used": "creative", "quality_score": 8.5},
    ]
    data = update_rewards(mock_results, "Tell me about gravity and write a poem")
    print_rewards(data)