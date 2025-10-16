# agents/decision_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json, os, re, time


# Minimal OpenAI-compatible client (env-driven)
@dataclass
class _LLMConfig:
    api_key: Optional[str]
    base_url: str
    model: str
    timeout: int

    @classmethod
    def from_env(cls) -> "_LLMConfig":
        def _clean(v: Optional[str], default: Optional[str] = None) -> str:
            if v is None or v == "":
                return (default or "").strip()
            v = v.strip().strip("'").strip('"')
            return v

        base = _clean(os.getenv("OPENAI_BASE_URL"), "https://api.openai.com/v1")
        if base and not re.match(r"^https?://", base, flags=re.I):
            base = "https://" + base.lstrip("/")
        base = base.rstrip("/")

        return cls(
            api_key=_clean(os.getenv("OPENAI_API_KEY")),
            base_url=base,
            model=_clean(os.getenv("OPENAI_MODEL"), "gpt-4.1-mini"),
            timeout=int(_clean(os.getenv("OPENAI_TIMEOUT"), "60") or "60"),
        )

    def ready(self) -> bool:
        return bool(self.api_key and self.base_url)


class _OpenAICompat:
    """Tiny /chat/completions JSON client (no external deps)."""
    def __init__(self, cfg: _LLMConfig) -> None:
        import urllib.request, urllib.error  # lazy import
        self._cfg = cfg
        self._http = urllib.request
        self._err = urllib.error

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        if not self._cfg.ready():
            raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_BASE_URL (DecisionAgent)")

        url = f"{self._cfg.base_url}/chat/completions"
        payload = {
            "model": self._cfg.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        req = self._http.Request(url, method="POST")
        req.add_header("Authorization", f"Bearer {self._cfg.api_key}")
        req.add_header("Content-Type", "application/json")
        data = json.dumps(payload).encode("utf-8")
        try:
            with self._http.urlopen(req, data=data, timeout=self._cfg.timeout) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            content = obj["choices"][0]["message"]["content"]
            return json.loads(content)
        except self._err.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI HTTP {e.code}: {body} (url={url})") from e
        except self._err.URLError as e:
            raise RuntimeError(f"OpenAI URL error: {e} (url={url})") from e
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e} (url={url})") from e


# =========================
# Decision Agent
# =========================

@dataclass
class DecisionConfig:
    use_llm: bool = True
    user_answer_with_rationale: bool = False
    max_context_chars: int = 4000   # per-doc; corpus uses ~4x this cap internally
    reflect: bool = False           # <-- NEW: enable reflection loop
    reflection_steps: int = 1       # simple single-step reflection by default


class DecisionAgent:
    """
    Answers high-level queries by combining results from other agents.
    Supports:
      - Per-document answers: ask(...) / answer_query(...)
      - Corpus answers across many docs: ask_corpus(...) / answer_corpus(...)
      - Reflection mode: reviews its own answer and refines it; logs trace.
    """

    def __init__(self, cfg: Optional[DecisionConfig] = None) -> None:
        self.cfg = cfg or DecisionConfig()
        self._llm_cfg = _LLMConfig.from_env()
        self._llm = _OpenAICompat(self._llm_cfg)

    # ---------- Public (per-document) ----------
    def ask(self, store, doc_id: str, question: str, *, plain: bool = True) -> str:
        out = self.answer_query(store, doc_id, question)
        if plain:
            return out.get("answer") or "I couldn't derive a confident answer."
        return json.dumps(out, ensure_ascii=False, indent=2)

    def answer_query(self, store, doc_id: str, question: str) -> Dict[str, Any]:
        context = self._gather_context(store, doc_id)
        if self.cfg.use_llm and self._llm_cfg.ready():
            try:
                result = self._answer_with_llm(context, question)
            except Exception as e:
                result = self._answer_heuristic(context, question)
                result["error"] = str(e)
        else:
            result = self._answer_heuristic(context, question)

        # ========== Reflection loop (per-doc) ==========
        if self.cfg.reflect:
            refl_log = self._reflect_result(store, [doc_id], context, question, result)
            # replace final answer with revised version
            revised = refl_log.get("final_revised_answer") or result.get("answer")
            result = {**result, "reflection": refl_log, "answer": revised}

        try:
            store.put_analysis(doc_id, "decision_single", result)
        except Exception as e:
            result["store_error"] = str(e)
        return result

    # ---------- Public (corpus) ----------
    def ask_corpus(self, store, doc_ids: List[str], question: str, *, plain: bool = True) -> str:
        out = self.answer_corpus(store, doc_ids, question)
        if plain:
            return out.get("answer") or "I couldn't derive a confident answer."
        return json.dumps(out, ensure_ascii=False, indent=2)

    def answer_corpus(self, store, doc_ids: List[str], question: str) -> Dict[str, Any]:
        merged = self._gather_corpus_context(store, doc_ids)
        if self.cfg.use_llm and self._llm_cfg.ready():
            try:
                result = self._answer_with_llm_corpus(merged, question)
            except Exception as e:
                result = self._answer_heuristic_corpus(merged, question)
                result["error"] = str(e)
        else:
            result = self._answer_heuristic_corpus(merged, question)

        # ========== Reflection loop (corpus) ==========
        if self.cfg.reflect:
            refl_log = self._reflect_result(store, doc_ids, merged, question, result)
            revised = refl_log.get("final_revised_answer") or result.get("answer")
            result = {**result, "reflection": refl_log, "answer": revised}

        # persist a single "corpus" decision row (doc_ids joined)
        try:
            store.put_analysis(",".join(doc_ids), "decision_corpus", result)
        except Exception as e:
            result["store_error"] = str(e)
        return result

    # ---------- Context gathering ----------
    def _gather_context(self, store, doc_id: str) -> Dict[str, Any]:
        doc = self._safe_get_document(store, doc_id)
        analyses = self._safe_get_analyses(store, doc_id)

        meta = (doc.get("meta") or {}) if doc else {}
        title = meta.get("title") or f"doc:{doc_id}"

        interesting: List[Dict[str, Any]] = []
        for row in analyses:
            task = (row.get("task") or "").lower()
            data = row.get("data") or row.get("result") or row
            if any(k in data for k in ("themes", "entities", "contradictions", "critical_issues", "patterns")):
                interesting.append({"task": task, "data": data, "analysis_id": row.get("id")})

        lines: List[str] = [f"[Title] {title}"]
        if meta.get("category"): lines.append(f"[Category] {meta['category']}")
        if meta.get("tone"): lines.append(f"[Tone] {meta['tone']}")
        kf = meta.get("key_findings") or []
        for k in kf[:5]:
            lines.append(f"[Finding] {k}")

        for row in interesting:
            data = row["data"]
            tname = row["task"] or "analysis"
            lines.append(f"[Task] {tname}")
            for k in ("themes", "entities", "patterns", "critical_issues", "contradictions", "follow_up_questions"):
                v = data.get(k)
                if isinstance(v, list) and v:
                    lines.append(f"[{k}] " + "; ".join(map(str, v))[:800])
                elif isinstance(v, str) and v:
                    lines.append(f"[{k}] {v}")

        context_text = "\n".join(lines)
        context_text = context_text[: self.cfg.max_context_chars]

        citations = []
        for row in interesting:
            if row.get("analysis_id") is not None:
                citations.append({
                    "analysis_id": row["analysis_id"],
                    "task": row["task"] or "analysis"
                })

        return {
            "doc_id": doc_id,
            "doc_meta": {"title": title},
            "context_text": context_text,
            "citations": citations[:12],
        }

    def _gather_corpus_context(self, store, doc_ids: List[str]) -> Dict[str, Any]:
        merged_lines: List[str] = []
        merged_cites: List[Dict[str, Any]] = []

        for did in doc_ids:
            ctx = self._gather_context(store, did)
            title = (ctx.get("doc_meta") or {}).get("title") or f"doc:{did}"
            merged_lines.append(f"=== {title} ({did}) ===")
            merged_lines.append(ctx.get("context_text", ""))
            for c in (ctx.get("citations") or []):
                merged_cites.append({"doc_id": did, "title": title, **c})

        context_text = "\n".join(merged_lines)
        cap = self.cfg.max_context_chars * 4
        if len(context_text) > cap:
            context_text = context_text[:cap]

        return {
            "context_text": context_text,
            "citations": merged_cites[:50],
            "timestamp": int(time.time() * 1000),
        }

    # ---------- LLM answers ----------
    def _answer_with_llm(self, ctx: Dict[str, Any], question: str) -> Dict[str, Any]:
        system = (
            "You are a Decision Agent. Answer a high-level question using only the supplied analyses. "
            "Return STRICT JSON with keys: "
            'answer (string), rationale (string), supporting_evidence (array of strings), '
            'citations (array of {"analysis_id": number, "task": string}), '
            'confidence ("low"|"medium"|"high").'
        )
        user = (
            f"Question:\n{question}\n\n"
            f"Context:\n{ctx.get('context_text','')}\n\n"
            "Citations:\n" + json.dumps(ctx.get("citations", []), ensure_ascii=False)
        )
        out = self._llm.chat_json(system, user)
        return self._normalize_single(out, ctx)

    def _answer_with_llm_corpus(self, source: Dict[str, Any], question: str) -> Dict[str, Any]:
        system = (
            "You are a senior Decision Agent. Combine analyses across multiple documents to answer a single high-level question. "
            "Return STRICT JSON with keys: "
            'answer (string), rationale (string), supporting_evidence (array of strings), '
            'citations (array of {"doc_id": string, "title": string, "analysis_id": number, "task": string}), '
            'confidence ("low"|"medium"|"high"). Base your answer ONLY on provided context.'
        )
        user = (
            f"User question:\n{question}\n\n"
            f"Combined context (from many docs):\n{source.get('context_text','')}\n\n"
            "Citations array:\n" + json.dumps(source.get("citations", []), ensure_ascii=False)
        )
        out = self._llm.chat_json(system, user)
        return self._normalize_corpus(out, source)

    # ---------- Heuristic answers (no LLM) ----------
    def _answer_heuristic(self, ctx: Dict[str, Any], question: str) -> Dict[str, Any]:
        text = ctx.get("context_text", "")
        has_contra = ("contradiction" in text.lower()) or ("[contradictions]" in text.lower())
        has_critical = any(w in text.lower() for w in ["risk", "gap", "limitation", "critical", "blocker"])
        lines = [f"{question.strip()} —"]
        if has_critical:
            lines.append("Address the highest-impact risks and gaps with clear owners and timelines.")
        if has_contra:
            lines.append("Resolve conflicting statements by selecting a source of truth and logging assumptions.")
        lines.append("Focus on themes that recur across analyses to maximize impact.")
        support = [l for l in text.split("\n") if l.startswith("[")]
        return {
            "answer": " ".join(lines),
            "rationale": "Heuristic fusion of analyzer outputs for a single document.",
            "supporting_evidence": support[:10],
            "citations": ctx.get("citations", [])[:10],
            "confidence": "medium" if (has_critical or has_contra) else "low",
        }

    def _answer_heuristic_corpus(self, source: Dict[str, Any], question: str) -> Dict[str, Any]:
        text = source.get("context_text", "")
        has_contra = ("contradiction" in text.lower()) or ("[contradictions]" in text.lower())
        has_critical = any(w in text.lower() for w in ["risk", "gap", "limitation", "critical", "blocker", "outage", "compliance"])
        lines = [f"{question.strip()} —"]
        if has_critical:
            lines.append("Across documents, multiple risks/critical issues emerge; prioritize mitigations with owners and deadlines.")
        if has_contra:
            lines.append("Cross-document contradictions exist; pick a source of truth and record decisions.")
        lines.append("Prioritize initiatives that align with the most repeated themes/entities across docs.")
        support = [l for l in text.split("\n") if l.startswith("[")]
        return {
            "answer": " ".join(lines),
            "rationale": "Heuristic fusion across documents (themes/patterns/contradictions/critical issues).",
            "supporting_evidence": support[:12],
            "citations": source.get("citations", [])[:12],
            "confidence": "medium" if (has_critical or has_contra) else "low",
        }

    # ---------- Reflection ----------
    def _reflect_result(
        self,
        store,
        doc_ids: List[str],
        source_ctx: Dict[str, Any],
        question: str,
        initial_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a reflection log:
          - original_answer
          - reflection_statement
          - final_revised_answer
        Persisted via store.put_analysis(doc_ids_joined, "decision_reflection", log)
        """
        original = (initial_result.get("answer") or "").strip()
        rationale = (initial_result.get("rationale") or "").strip()
        confidence = (initial_result.get("confidence") or "").lower()

        reflection: str
        revised: str

        if self.cfg.use_llm and self._llm_cfg.ready():
            system = (
                "You are a rigorous reviewer. Reflect on the agent's original answer, identify gaps or missed angles, "
                "and then provide a concise revised answer. Return STRICT JSON with keys: "
                'reflection (string), revised_answer (string). Keep the revised answer actionable and grounded only in the provided context.'
            )
            user = (
                f"Question:\n{question}\n\n"
                f"Original answer:\n{original}\n\n"
                f"Original rationale:\n{rationale}\n\n"
                f"Confidence: {confidence or 'n/a'}\n\n"
                f"Context (evidence to stay grounded):\n{source_ctx.get('context_text','')}"
            )
            try:
                out = self._llm.chat_json(system, user)
                reflection = (out.get("reflection") or "").strip()
                revised = (out.get("revised_answer") or original).strip()
            except Exception as e:
                reflection = f"Review fallback: unable to call LLM ({e}). Emphasize risks/contradictions and specificity."
                revised = self._heuristic_revision(original, source_ctx)
        else:
            # Heuristic reflection
            reflection = self._heuristic_reflection(original, source_ctx, confidence)
            revised = self._heuristic_revision(original, source_ctx)

        log = {
            "original_answer": original,
            "reflection_statement": reflection,
            "final_revised_answer": revised,
            "question": question,
            "timestamp": int(time.time() * 1000),
            "doc_ids": doc_ids,
        }

        # persist reflection log
        try:
            store.put_analysis(",".join(doc_ids), "decision_reflection", log)
        except Exception:
            pass

        return log

    def _heuristic_reflection(self, original: str, source_ctx: Dict[str, Any], confidence: str) -> str:
        low = source_ctx.get("context_text", "").lower()
        flags = []
        if "contradiction" in low or "[contradictions]" in low:
            flags.append("possible cross-source contradictions not resolved")
        if any(w in low for w in ["risk", "gap", "limitation", "blocker", "outage", "compliance"]):
            flags.append("risks/limitations may lack owners or timelines")
        if len(original) < 60:
            flags.append("answer might be too brief or generic")
        if confidence in ("low", ""):
            flags.append("low confidence — consider citing evidence more directly")
        if not flags:
            return "No major issues detected; tightened phrasing and added prioritization."
        return "I may have missed or underemphasized: " + "; ".join(flags) + "."

    def _heuristic_revision(self, original: str, source_ctx: Dict[str, Any]) -> str:
        text = source_ctx.get("context_text", "")
        # Try to pull a couple of evidence lines to ground the revision
        evid = [l for l in text.split("\n") if l.startswith("[Finding]") or l.startswith("[critical_issues]")]
        hint = (" Prioritize top risks and resolve contradictions with named owners and deadlines."
                " Cite the most repeated themes to justify prioritization.")
        core = original if original else "Prioritize high-impact risks and reconcile contradictions."
        if evid:
            ev = "; ".join(evid[:2])[:300]
            return f"{core} Grounding evidence: {ev}.{hint}"
        return core + hint

    # ---------- Normalizers ----------
    def _normalize_single(self, out: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        def _as_list(v):
            if v is None: return []
            if isinstance(v, list): return [str(x) if not isinstance(x, dict) else x for x in v]
            if isinstance(v, str) and v.strip(): return [v.strip()]
            return []
        cites_in = out.get("citations") or []
        norm_cites = []
        for c in cites_in:
            if isinstance(c, dict) and all(k in c for k in ("analysis_id","task")):
                norm_cites.append({"analysis_id": c["analysis_id"], "task": str(c["task"])})
        return {
            "answer": (out.get("answer") or "").strip(),
            "rationale": (out.get("rationale") or "").strip(),
            "supporting_evidence": _as_list(out.get("supporting_evidence"))[:12],
            "citations": norm_cites[:12] if norm_cites else ctx.get("citations", [])[:12],
            "confidence": (out.get("confidence") or "medium").lower() if isinstance(out.get("confidence"), str) else "medium",
        }

    def _normalize_corpus(self, out: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        def _as_list(v):
            if v is None: return []
            if isinstance(v, list): return [str(x) if not isinstance(x, dict) else x for x in v]
            if isinstance(v, str) and v.strip(): return [v.strip()]
            return []
        cites_in = out.get("citations") or []
        norm_cites = []
        for c in cites_in:
            if isinstance(c, dict) and all(k in c for k in ("doc_id","title","analysis_id","task")):
                norm_cites.append({
                    "doc_id": str(c["doc_id"]),
                    "title": str(c["title"]),
                    "analysis_id": c["analysis_id"],
                    "task": str(c["task"]),
                })
        conf = (out.get("confidence") or "medium").lower() if isinstance(out.get("confidence"), str) else "medium"
        if conf not in {"low","medium","high"}:
            conf = "medium"
        return {
            "answer": (out.get("answer") or "").strip(),
            "rationale": (out.get("rationale") or "").strip(),
            "supporting_evidence": _as_list(out.get("supporting_evidence"))[:12],
            "citations": norm_cites[:12] if norm_cites else source.get("citations", [])[:12],
            "confidence": conf,
        }

    # ---------- Store helpers ----------
    def _safe_get_document(self, store, doc_id: str) -> Dict[str, Any]:
        try:
            return store.get_document(doc_id) or {}
        except Exception:
            return {}

    def _safe_get_analyses(self, store, doc_id: str) -> List[Dict[str, Any]]:
        for meth in ("get_analyses", "list_analyses", "find_analyses"):
            if hasattr(store, meth):
                try:
                    rows = getattr(store, meth)(doc_id) or []
                    if isinstance(rows, list):
                        return rows
                except Exception:
                    pass
        return []
