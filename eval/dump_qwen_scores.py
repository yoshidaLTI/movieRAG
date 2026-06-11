import json

PATH  = "claude_knowledge/RAG-EVAL_QWEN-3-VL30b.json"
MODEL = "qwen3-vl:30b"

with open(PATH, encoding="utf-8") as f:
    data = json.load(f)

questions = sorted(data["questions"], key=lambda q: q["qg_id"])

for q in questions:
    score = q.get("accuracy_by_model", {}).get(MODEL)
    print(score)
