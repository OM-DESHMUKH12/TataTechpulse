"""Run a transparent, deterministic retrieval test against the included synthetic dataset."""
from pathlib import Path
import csv, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import extract_text_file, split_pages, retrieve

ROOT = Path(__file__).resolve().parents[1]
kb_path = ROOT / "Input_Data" / "synthetic_autosar_hld.txt"
questions_path = ROOT / "Input_Data" / "evaluation_questions.csv"
out_dir = ROOT / "Evaluation_Results"
out_dir.mkdir(exist_ok=True)

pages = extract_text_file(kb_path.read_bytes(), kb_path.name)
chunks = split_pages(pages)
rows = []
with questions_path.open(encoding="utf-8-sig", newline="") as f:
    for item in csv.DictReader(f):
        results = retrieve(item["question"], chunks, top_k=3)
        evidence = " ".join(r["text"].lower() for r in results)
        expected = [x.strip().lower() for x in item["expected_terms"].split("|") if x.strip()]
        hits = [term for term in expected if term in evidence]
        passed = bool(expected) and len(hits) / len(expected) >= 0.5
        rows.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "expected_terms": "|".join(expected),
            "matched_terms": "|".join(hits),
            "term_coverage": round(len(hits) / len(expected), 3) if expected else 0,
            "retrieval_test_pass": passed,
            "top_source": results[0]["source"] if results else "",
            "top_page": results[0]["page"] if results else "",
            "top_score": round(results[0]["score"], 4) if results else 0,
        })

csv_path = out_dir / "retrieval_evaluation.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)

summary = {
    "dataset": kb_path.name,
    "dataset_type": "synthetic illustrative HLD; not OEM or normative AUTOSAR content",
    "retrieval_method": "TF-IDF word/bigram features with cosine similarity",
    "top_k": 3,
    "question_count": len(rows),
    "passed": sum(1 for r in rows if r["retrieval_test_pass"]),
    "pass_rate": round(sum(1 for r in rows if r["retrieval_test_pass"]) / len(rows), 3) if rows else 0,
    "metric_definition": "A question passes if at least 50% of its manually specified expected evidence terms appear in the top-3 retrieved text.",
    "warning": "This is a small lexical retrieval smoke test. It does not evaluate answer factuality, semantic relevance robustly, safety, or standards compliance.",
    "results_file": csv_path.name,
}
(out_dir / "retrieval_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"\nDetailed results: {csv_path}")
