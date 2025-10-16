# agents/analyzer_agent_openai.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import os
import requests

try:
    from Core.memory_store import MemoryStore  
except Exception:
    MemoryStore = object  # fallback for type hints

# OpenAI-compatible minimal client (env-only)
@dataclass
class LLMConfig:
    api_key: Optional[str]
    base_url: str
    model: str
    timeout: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.avalai.ir/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
        )

    def ensure_ready(self) -> None:
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY (set in environment or .env).")


class OpenAICompat:
    #OpenAI /chat/completions client using urllib
    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        self._cfg.ensure_ready()
        url = self._cfg.base_url.rstrip("/") + "/chat/completions"

        payload = {
            "model": self._cfg.model,
            "temperature": 0.2, 
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,                   
                timeout=self._cfg.timeout
            )
            resp.raise_for_status()
            obj = resp.json()
            content = obj["choices"][0]["message"]["content"]
            return json.loads(content)
        except requests.HTTPError as e:
            # Include server body for debugging
            body = getattr(e.response, "text", "")
            code = getattr(e.response, "status_code", "unknown")
            raise RuntimeError(f"OpenAI HTTP {code}: {body}") from e
        except requests.RequestException as e:
            # Network / timeout / connection errors
            raise RuntimeError(f"OpenAI request failed: {e}") from e
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            # Malformed or unexpected payload
            raw = resp.text if 'resp' in locals() else ''
            raise RuntimeError(f"OpenAI response parsing failed: {e}. Raw: {raw}") from e

# Analyzer (env-only) with shared-memory helpers
@dataclass
class AnalyzerConfig: #adapt as needed
    model_task: str = "general_analysis"  # or "scientific_analysis", "policy_summary"
    excerpt_chars: int = 1500
    include_key_findings: bool = True

class OpenAIAnalyzerAgent:
    """
    Analyzer using an OpenAI-compatible /chat/completions endpoint with JSON mode.
    Reads credentials/config from environment variables ONLY.
    Extracts:
      - themes: string[]
      - entities: string[]
      - sentiment: "positive" | "neutral" | "negative"
      - contradictions: string[]
      - patterns: string[]                
      - critical_issues: string[]         
      - follow_up_questions: string[]
    """

    def __init__(self, cfg: Optional[AnalyzerConfig] = None) -> None:
        self.cfg = cfg or AnalyzerConfig()
        self.llm_cfg = LLMConfig.from_env()
        self.llm = OpenAICompat(self.llm_cfg)

    # Prompt helpers (Considered these 3 as the main 3 categories)
    def _task_instruction(self, task: str) -> str:
        mapping = {
            "general_analysis": (
                "Identify 3–5 concise themes (noun phrases), extract up to 10 salient entities, "
                "set overall sentiment to one of [positive, neutral, negative], list contradictions (empty if none), "
                "surface recurring patterns (repeated motifs, consistent claims/concerns), "
                "flag critical issues (gaps, risks, inconsistencies, blockers, anomalies), "
                "and propose 1–3 follow_up_questions."
            ),
            "scientific_analysis": (
                "Identify 3–5 scientific themes, extract scientific entities (e.g., neurotransmitters, bacteria, methods), "
                "set sentiment [positive, neutral, negative], list contradictions (empty if none), "
                "surface recurring patterns in findings/methods, "
                "flag critical issues (methodological gaps, confounds, biases, unclear assumptions), "
                "and propose 1–3 next-step research questions."
            ),
            "policy_summary": (
                "Extract policy domain, key stakeholders, main objectives, risks, "
                "surface recurring patterns (systemic issues, repeated constraints), "
                "flag critical issues (compliance gaps, high-risk items, blockers), "
                "and propose 1–2 actionable follow_up_questions; set sentiment [positive, neutral, negative]."
            ),
        }
        return mapping.get(task, mapping["general_analysis"])

    def _build_user_prompt(self, reader_output: Dict[str, Any], task: str) -> str:
        text = reader_output.get("text") or reader_output.get("summary") or ""
        meta = reader_output.get("meta", {})
        meta_lines = [f"{k}: {v}" for k, v in meta.items() if v]
        meta_block = "\n".join(meta_lines)

        # Strict JSON schema with required fields
        return (
            "Analyze the document below and return ONLY valid JSON that conforms to the schema. "
            "If a field has no content, return an empty array; for sentiment, return one of "
            '["positive","neutral","negative"].\n\n'
            f"Metadata:\n{meta_block}\n\n"
            "Document:\n"
            f"{text}\n\n"
            "JSON schema (keys only; fill with concrete values):\n"
            "{\n"
            '  "themes": [],\n'
            '  "entities": [],\n'
            '  "sentiment": "",\n'
            '  "contradictions": [],\n'
            '  "patterns": [],\n'
            '  "critical_issues": [],\n'
            '  "follow_up_questions": []\n'
            "}\n"
            "Return strictly JSON — no text outside the JSON object."
        )

    # Main analysis on provided payload (Data + prompt)
    def analyze(self, reader_output: Dict[str, Any], task: Optional[str] = None) -> Dict[str, Any]:
        task = task or self.cfg.model_task
        system = (
            "You are a precise analytical assistant. "
            "Respond with STRICT JSON only. Required keys: themes (array of strings), entities (array of strings), "
            "sentiment (string: positive|neutral|negative), contradictions (array of strings), "
            "patterns (array of strings), critical_issues (array of strings), "
            "follow_up_questions (array of strings). "
            "Do not include any text outside the JSON."
        )
        user = self._build_user_prompt(reader_output, task) + "\nTask details: " + self._task_instruction(task)

        try:
            out = self.llm.chat_json(system, user)
        except Exception as e:
            print(e)
            return {
                "themes": [],
                "entities": [],
                "sentiment": "neutral",
                "contradictions": [],
                "patterns": [],
                "critical_issues": [],
                "follow_up_questions": [],
                "error": str(e),
            }

        # Normalize and guarantee keys
        def _as_list(v):
            if v is None: return []
            if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
            if isinstance(v, str) and v.strip(): return [v.strip()]
            return []

        result = {
            "themes": _as_list(out.get("themes")),
            "entities": _as_list(out.get("entities")),
            "sentiment": (out.get("sentiment") or "neutral").strip().lower(),
            "contradictions": _as_list(out.get("contradictions")),
            "patterns": _as_list(out.get("patterns")),
            "critical_issues": _as_list(out.get("critical_issues")),
            "follow_up_questions": _as_list(out.get("follow_up_questions")),
        }
        if result["sentiment"] not in {"positive", "neutral", "negative"}:
            result["sentiment"] = "neutral"
        return result

    # Store-aware helpers 
    def _compact_from_stored_doc(self, stored_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Build the compact reader_output expected by analyze(), from a stored ParsedDocument row."""
        meta = stored_doc.get("meta", {}) or {}
        sections = stored_doc.get("sections", {}) or {}
        keys = list(sections.keys())
        preferred = [k for k in keys if any(w in k.lower() for w in ("abstract", "intro", "background"))]
        first_key = preferred[0] if preferred else (keys[0] if keys else None)
        excerpt = sections.get(first_key, "")[: self.cfg.excerpt_chars] if first_key else ""

        lines = [f"Title: {meta.get('title', '')}"]
        if meta.get("category"): lines.append(f"Category: {meta.get('category')}")
        if meta.get("tone"): lines.append(f"Tone: {meta.get('tone')}")
        kfs = meta.get("key_findings")
        if self.cfg.include_key_findings and isinstance(kfs, list) and kfs:
            lines.append("Key findings:")
            for k in kfs[:5]:
                lines.append(f"- {k}")
        lines.append("Excerpt:")
        lines.append(excerpt)
        compact = "\n".join(lines)

        return {
            "text": compact,
            "meta": {
                "title": meta.get("title"),
                "category": meta.get("category"),
                "tone": meta.get("tone"),
                "key_findings": kfs or [],
            },
        }

    def analyze_doc(self, store: MemoryStore, doc_id: str, task: Optional[str] = None) -> Dict[str, Any]:
        # Loads the parsed document from the shared store, analyze it, and
        # persists the analysis back to the store. Returns the analysis dict.
        stored_doc = store.get_document(doc_id)
        if not stored_doc:
            return {
                "themes": [],
                "entities": [],
                "sentiment": "neutral",
                "contradictions": [],
                "patterns": [],
                "critical_issues": [],
                "follow_up_questions": [],
                "error": f"Document not found in store: {doc_id}",
            }

        payload = self._compact_from_stored_doc(stored_doc)
        result = self.analyze(payload, task=task or self.cfg.model_task)

        # Save back to store (any JSON is accepted in store)
        try:
            store.put_analysis(doc_id, task or self.cfg.model_task, result)
        except Exception as e:
            result = {**result, "store_error": str(e)}

        return result

    async def analyze_async(self, reader_output: Dict[str, Any], task: Optional[str] = None) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze, reader_output, task)
