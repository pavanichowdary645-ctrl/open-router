import json
from groq import Groq
from shared_config import GROQ_KEY, AnalyzerOutput

client = Groq(api_key=GROQ_KEY)

SYSTEM_PROMPT = """
You are a prompt analyzer. Given a user prompt, return ONLY valid JSON with this exact structure:
{
  "complexity": "simple" | "medium" | "complex",
  "segments": [
    {"text": "<segment text>", "type": "creative" | "math" | "general" | "coding"}
  ]
}

Rules:
- complexity = simple if it is a basic factual or conversational question
- complexity = medium if it needs moderate reasoning or mixed tasks
- complexity = complex if it requires deep reasoning, multi-step, or highly technical work
- Split the prompt into logical segments by task type
- If the whole prompt is one type, return one segment
- Return ONLY the JSON, no explanation, no markdown fences
"""

def run_analyzer(user_prompt: str) -> AnalyzerOutput:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result: AnalyzerOutput = json.loads(raw.strip())
    return result

if __name__ == "__main__":
    test_prompt = "Explain how gravity works and write a Python function to calculate gravitational force"
    out = run_analyzer(test_prompt)
    print(json.dumps(out, indent=2))