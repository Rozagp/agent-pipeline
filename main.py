# main.py
from pathlib import Path
import os
import json
import argparse
import random
import time

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from core.memory_store import make_store
from agents.reader_agent import ReaderAgent
from agents.analyzer_agent import OpenAIAnalyzerAgent, AnalyzerConfig
from agents.decision_agent import DecisionAgent, DecisionConfig

SUPPORTED_EXTS = (".txt", ".md")

def iter_doc_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])

def main():
    ap = argparse.ArgumentParser(description="Reader → Analyzer → Decision pipeline (multi-file + corpus decision)")
    ap.add_argument("--dir", type=str, default="docs", help="Directory containing .txt/.md files")
    ap.add_argument("--task", type=str, default="scientific_analysis",
                    choices=["scientific_analysis", "general_analysis", "policy_summary"])
    ap.add_argument("--q", type=str, default=None, help="Per-document question for Decision Agent")
    ap.add_argument("--plain", action="store_true", help="Per-document answer as user-friendly text")
    ap.add_argument("--offline", action="store_true", help="Force analyzer heuristic mode (no LLM calls)")
    ap.add_argument("--global-q", type=str, default=None, help="One high-level question across ALL processed docs")
    ap.add_argument("--global-plain", action="store_true", help="Return user-friendly text for the global decision")
    ap.add_argument("--reflect", action="store_true", help="Enable Decision Agent reflection and revision")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between docs (throttling).")
    ap.add_argument("--jitter", type=float, default=0.0, help="Optional ±random jitter (seconds) added to --sleep.")
    args = ap.parse_args()

    files = iter_doc_files(Path(args.dir))
    if not files:
        print(f"No .txt/.md files found under: {Path(args.dir).resolve()}")
        return

    store = make_store(backend=os.getenv("STORE_BACKEND", "sqlite"))
    reader = ReaderAgent()
    if args.offline:
        os.environ["ANALYZER_OFFLINE"] = "1"
    else:
        os.environ.pop("ANALYZER_OFFLINE", None)
    analyzer = OpenAIAnalyzerAgent(cfg=AnalyzerConfig(model_task=args.task))
    decision = DecisionAgent(
    DecisionConfig(
        use_llm=not args.offline,
        user_answer_with_rationale=False,
        reflect=args.reflect,            
        reflection_steps=1
    ))

    processed_doc_ids: list[str] = []

    for idx, path in enumerate(files, start=1):
        print("\n" + "=" * 60)
        print(f"Document {idx}/{len(files)}: {path}")
        print("=" * 60)

        # Reader → store
        doc = reader.parse(path, use_llm=False)
        doc_id = store.put_document(doc.model_dump())
        processed_doc_ids.append(doc_id)

        print("\n--- Reader (stored) ---")
        print(json.dumps({"doc_id": doc_id, "title": doc.meta.title}, ensure_ascii=False, indent=2))

        # Analyzer → store
        analysis = analyzer.analyze_doc(store, doc_id, task=args.task)
        print("\n--- Analyzer Output (stored) ---")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        
        # Per-document Decision (optional)
        per_doc_q = args.q or "Given the analyses, what are the key risks, contradictions, and next recommended steps?"
        if args.plain:
            print("\n--- Decision (user-friendly) ---")
            print(decision.ask(store, doc_id, per_doc_q, plain=True))
        else:
            print("\n--- Decision (structured) ---")
            print(json.dumps(decision.answer_query(store, doc_id, per_doc_q), ensure_ascii=False, indent=2))
        if idx < len(files) and (args.sleep > 0 or args.jitter > 0):
            base = args.sleep
            jit = random.uniform(-args.jitter, args.jitter) if args.jitter > 0 else 0.0
            delay = max(0.0, base + jit)
            print(f"\n[throttle] Sleeping {delay:.2f}s to avoid rate limits...")
            time.sleep(delay)

    # Global (corpus) Decision across all processed docs
    if args.global_q and processed_doc_ids:
        print("\n" + "=" * 60)
        print("Global Decision Across All Documents")
        print("=" * 60)
        if args.global_plain:
            print(decision.ask_corpus(store, processed_doc_ids, args.global_q, plain=True))
        else:
            print(json.dumps(decision.answer_corpus(store, processed_doc_ids, args.global_q),
                             ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
