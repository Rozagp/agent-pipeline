# tests/test_analyzer.py
from agents.reader_agent import ReaderAgent
from agents.analyzer_agent import OpenAIAnalyzerAgent, AnalyzerConfig

def test_analyzer_offline_extracts_fields(tmp_path, store):
    p = tmp_path / "doc.txt"
    p.write_text(
        "Engineering Report\n\n"
        "Risks:\n"
        "- Critical dependency outage could block delivery.\n"
        "Contradictions:\n"
        "- Metric A is 95% here but elsewhere listed as 88%.\n",
        encoding="utf-8",
    )

    reader = ReaderAgent()
    parsed = reader.parse(p, use_llm=False)
    doc_id = store.put_document(parsed.model_dump())

    analyzer = OpenAIAnalyzerAgent(cfg=AnalyzerConfig(model_task="scientific_analysis"))
    result = analyzer.analyze_doc(store, doc_id, task="scientific_analysis")

    assert isinstance(result, dict)
    for key in ["themes", "entities", "sentiment", "contradictions", "patterns", "critical_issues"]:
        assert key in result
