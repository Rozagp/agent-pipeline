from pathlib import Path
from agents.reader_agent import ReaderAgent

def test_reader_parses_basic(tmp_path: Path):
    p = tmp_path / "sample.md"
    p.write_text(
        "---\n"
        'title: "Demo Doc"\n'
        "---\n\n"
        "# Intro\n"
        "Hello world.\n\n"
        "## Findings\n"
        "- This is a fairly long bullet point that should qualify as a key finding because it exceeds 20 characters.\n"
        "- Another sufficiently long key finding for tests.\n",
        encoding="utf-8",
    )

    agent = ReaderAgent()
    doc = agent.parse(p, use_llm=False)

    assert doc.meta.title == "Demo Doc"
    assert len(doc.meta.key_findings) >= 2
    assert any(k.startswith("## Findings") for k in doc.sections.keys())
    assert doc.tokens > 0
