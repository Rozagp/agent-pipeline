from agents.decision_agent import DecisionAgent, DecisionConfig

def _base_doc(doc_id: str, title: str, findings):
    return {
        "id": doc_id,
        "meta": {"title": title, "key_findings": findings, "tone": "neutral", "category": "engineering"},
        "sections": {"# Intro": "hello", "## Risks": "- outage\n- compliance gap"},
        "tokens": 100,
        "path": f"docs/{doc_id}.md",
    }

def test_decision_corpus_with_reflection(store):
    # Two docs in the store
    d1 = _base_doc("a1", "Service A Report", ["SLO breaches increased", "Dependency X is unstable"])
    d2 = _base_doc("b2", "Service B Report", ["Error rate inconsistent", "Conflicting SLO numbers in docs"])
    store.put_document(d1)
    store.put_document(d2)

    # Fake analyzer outputs the Decision Agent will read
    store.put_analysis("a1", "scientific_analysis", {
        "themes": ["reliability", "dependencies"],
        "entities": ["Service A", "Dependency X"],
        "contradictions": [],
        "critical_issues": ["SLO breaches"],
        "patterns": ["recurring incidents"],
        "follow_up_questions": [],
        "sentiment": "cautious",
    })
    store.put_analysis("b2", "scientific_analysis", {
        "themes": ["reliability", "metrics"],
        "entities": ["Service B"],
        "contradictions": ["SLO stated as 95% vs 88%"],
        "critical_issues": ["metric inconsistency"],
        "patterns": ["spiky error rate"],
        "follow_up_questions": [],
        "sentiment": "cautious",
    })

    # Decision across corpus, offline + reflection
    decision = DecisionAgent(DecisionConfig(use_llm=False, reflect=True))
    out = decision.answer_corpus(store, ["a1", "b2"], "Where are the biggest risks and contradictions across the corpus?")

    assert "answer" in out and isinstance(out["answer"], str)
    assert "citations" in out and isinstance(out["citations"], list)
    assert "reflection" in out
    refl = out["reflection"]
    for k in ["original_answer", "reflection_statement", "final_revised_answer"]:
        assert k in refl and isinstance(refl[k], str)
