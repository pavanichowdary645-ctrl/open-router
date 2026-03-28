import os
from typing import TypedDict, List, Literal

_BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))

GROQ_KEY = os.getenv("GROQ_KEY", "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")

MODELS = {
    "analyzer":  "llama-3.3-70b-versatile",
    "combiner":  "llama-3.3-70b-versatile",
    "simple":    "llama-3.1-8b-instant",
    "math":      "openai/gpt-oss-120b",
    "creative":  "llama-3.3-70b-versatile",
    "coding":    "qwen/qwen3-32b",
}

# Cost per 1000 tokens in USD
MODEL_COST_PER_1K_TOKENS = {
    "llama-3.3-70b-versatile": 0.00059,
    "llama-3.1-8b-instant":    0.00005,
    "openai/gpt-oss-120b":     0.00015,
    "qwen/qwen3-32b":          0.00029,
}

# Quality score out of 10
MODEL_QUALITY_SCORE = {
    "llama-3.3-70b-versatile": 8.5,
    "llama-3.1-8b-instant":    6.5,
    "openai/gpt-oss-120b":     9.5,
    "qwen/qwen3-32b":          8.8,
}

# Baseline: most expensive model cost for reward calculation
EXPENSIVE_MODEL_COST_PER_1K = 0.00059

OUTPUT_DIR     = os.path.join(_BACKEND_ROOT, "outputs") + os.sep
ANALYZER_OUT   = OUTPUT_DIR + "analyzer_output.json"
DECISION_OUT   = OUTPUT_DIR + "decision_output.json"
EXECUTION_OUT  = OUTPUT_DIR + "execution_output.json"
FINAL_OUT      = OUTPUT_DIR + "final_output.json"
METRICS_LOG    = OUTPUT_DIR + "metrics.json"
ROUTING_MEMORY = OUTPUT_DIR + "routing_memory.json"
CREDITS_FILE   = OUTPUT_DIR + "credits.json"

class Segment(TypedDict):
    text: str
    type: Literal["creative", "math", "general", "coding"]

class AnalyzerOutput(TypedDict):
    complexity: Literal["simple", "medium", "complex"]
    segments: List[Segment]

class DecisionItem(TypedDict):
    text: str
    model: str
    params: dict

class DecisionOutput(TypedDict):
    items: List[DecisionItem]

class ExecutionItem(TypedDict):
    segment_text:   str
    model_used:     str
    model_name:     str
    params_used:    dict
    response:       str
    latency_ms:     float
    tokens_used:    int
    cost_usd:       float
    quality_score:  float

class ExecutionOutput(TypedDict):
    results: List[ExecutionItem]

class FinalOutput(TypedDict):
    original_prompt:   str
    combined_response: str
    metrics:           dict