"""Evaluation. Needs two files:
  qa.csv       columns: question,expected_company   (20 rows, written from your company documents)
  tool_qa.csv  columns: question,expected_route     (10 rows; route is stock, currency or datetime)
Also saves eval_answers.csv so you can grade correctness by hand against the source documents."""
import re, time, uuid
import pandas as pd
from agent import chat, ask, MODEL


def number(raw: str) -> float:
    m = re.search(r"\d(?:\.\d+)?", raw)
    return min(float(m.group(0)), 1.0) if m else 0.0


qa = pd.read_csv("qa.csv")
hits, scores, rel, times, answers = [], [], [], [], []
for _, row in qa.iterrows():
    t = time.time()
    r = chat(row["question"], str(uuid.uuid4()))
    times.append(time.time() - t)
    scores.append(r["score"])
    hits.append(row["expected_company"] in r["sources"])                 # retrieval hit@3
    rel.append(number(ask("Rate from 0.0 to 1.0 how directly the ANSWER addresses the QUESTION. Number only.",
                          f"QUESTION: {row['question']}\nANSWER: {r['answer']}")))
    answers.append(r["answer"])
qa["answer"], qa["hit_at_3"], qa["faithfulness"] = answers, hits, scores
qa["correct_by_hand"] = ""                                               # fill in Y / N yourself
qa.to_csv("eval_answers.csv", index=False)

tool = pd.read_csv("tool_qa.csv")
ok = sum(chat(r.question, str(uuid.uuid4()))["route"] == r.expected_route for r in tool.itertuples())

n = len(qa)
print(f"Company questions:      {n}")
print(f"Faithfulness (avg):     {sum(scores)/n:.2f}")
print(f"Answer relevancy (avg): {sum(rel)/n:.2f}   (LLM judge: {MODEL})")
print(f"Retrieval hit@3:        {sum(hits)/n:.2f}")
print(f"Avg response time:      {sum(times)/n:.1f} s")
print(f"Tool-call accuracy:     {ok} / {len(tool)}")
print("Now open eval_answers.csv and mark correct_by_hand (Y/N) for the correctness criterion.")
