"""
reader Agent: Parses .txt/.md documents, extract metadata and sections, and
produces a concise JSON summary.

Supports LLM-enhanced extraction via an OpenAI-compatible endpoint.

Usage:
    # Heuristic only (no LLM)
    python -m agents.reader_agent path/to/doc.md (or .txt)  --json

    # LLM-enhanced (requires OPENAI_* env)
    python -m agents.reader_agent path/to/doc.md (or .txt) --json --use-llm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from agents.LLMconfig import LLMConfig
from agents.LLMconfig import OpenAICompatLLM


# Data Models


class DocMeta(BaseModel):
    title: str = Field(..., description="Detected document title")
    key_findings: List[str] = Field(default_factory=list, description="3–7 concise bullets")
    tone: Optional[str] = Field(default=None, description="Single-word tone estimate")
    category: Optional[str] = Field(default=None, description="Short category label")

class ParsedDocument(BaseModel):
    id: str
    path: str
    tokens: int
    meta: DocMeta
    sections: Dict[str, str]

# OpenAI-compatible LLM (optional)

    

@dataclass
class ReaderConfig:
    min_finding_len: int = 20
    max_findings: int = 7
    min_findings: int = 3

class ReaderAgent:
    """
    Pipeline:
      1) Load & normalize text
      2) Strip frontmatter & HTML
      3) Detect title, split sections (heuristics)
      4) Extract findings + tone/category (heuristics)
      5) If LLM configured and `use_llm=True`, refine meta via LLM JSON output
    """
    def __init__(self, config: Optional[ReaderConfig] = None, llm_cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = config or ReaderConfig()
        self.llm_cfg = llm_cfg or LLMConfig.from_env()
        self.llm = OpenAICompatLLM(self.llm_cfg) if self.llm_cfg.enabled() else None

    # Public API 
    def parse(self, path: Path, *, use_llm: bool = False) -> ParsedDocument:
        text = self._load_text(path)
        text = self._normalize_newlines(text)
        body, fm_meta = self._strip_frontmatter(text)
        plain = self._strip_html(body)

        title = (
            fm_meta.get("title")
            or self._first_markdown_h1(body)
            or self._first_nonempty_line(body)
            or path.stem.replace("_", " ").replace("-", " ").strip()
        )

        sections = self._split_sections(body) or {"Body": body.strip()}

        key_findings = self._extract_key_findings(plain, sections)
        tone, category = self._guess_tone_category(plain, fm_meta)

        meta = DocMeta(title=title.strip(), key_findings=key_findings, tone=tone, category=category)

        # Optionally refine with LLM
        if use_llm and self.llm and self.llm_cfg.enabled():
            system = (
                "You are a careful document analyst. Return STRICT JSON with keys: "
                "title (string), key_findings (array of 3-7 strings, each >= 20 chars), "
                "tone (one word), category (1-3 words). No extra keys."
            )
            user = (
                "Read the following document and extract fields. If title is missing, "
                f"use this fallback title: '{meta.title}'.\n\n" + plain
            )
            try:
                parsed = self.llm.chat_json(system, user)
                title = str(parsed.get("title") or meta.title).strip()
                kf = [str(x).strip() for x in (parsed.get("key_findings") or []) if str(x).strip()]
                tone = parsed.get("tone") or meta.tone
                category = parsed.get("category") or meta.category

                # keep same logic for merging findings
                merged_kf = kf[: self.cfg.max_findings]
                if len(merged_kf) < self.cfg.min_findings:
                    for k in meta.key_findings:
                        if len(merged_kf) >= self.cfg.min_findings:
                            break
                        if k not in merged_kf:
                            merged_kf.append(k)

                meta = DocMeta(
                    title=title or meta.title,
                    key_findings=merged_kf[: self.cfg.max_findings],
                    tone=tone,
                    category=category,
                )
            except Exception as e:  # pragma: no cover
                print(f"[LLM] Extraction failed: {e}")


        doc_id = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
        tokens = self._rough_token_count(plain)
        return ParsedDocument(id=doc_id, path=str(path), tokens=tokens, meta=meta, sections=sections)

    # Loading & cleaning
    def _load_text(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="ignore")

    def _normalize_newlines(self, s: str) -> str:
        return s.replace("\r\n", "\n").replace("\r", "\n")

    _FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

    def _strip_frontmatter(self, text: str) -> Tuple[str, Dict[str, str]]:
        m = self._FRONTMATTER_RE.match(text)
        if not m:
            return text, {}
        fm = m.group(1)
        body = text[m.end() :]
        meta: Dict[str, str] = {}
        for line in fm.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip().strip('"\'')
        return body, meta

    _HTML_TAG_RE = re.compile(r"<[^>]+>")

    def _strip_html(self, text: str) -> str:
        return self._HTML_TAG_RE.sub(" ", text)

    # Titles & sections 
    _H1_RE = re.compile(r"#^\s+(.+)$", re.MULTILINE)

    def _first_markdown_h1(self, text: str) -> Optional[str]:
        m = self._H1_RE.search(text)
        return m.group(1).strip() if m else None

    def _first_nonempty_line(self, text: str) -> Optional[str]:
        for line in text.split("\n"):
            t = line.strip()
            if t:
                return t
        return None

    def _split_sections(self, text: str) -> Dict[str, str]:
        lines = text.split("\n")
        sections: Dict[str, List[str]] = {}
        current = "Intro"
        sections[current] = []

        heading_md = re.compile(r"^(#+)\s+(.*)$")
        heading_num = re.compile(r"^\s*(\d+)\.\s+([A-Z][^\n]{2,})$")  # e.g., "1. Abstract"

        for ln in lines:
            m1 = heading_md.match(ln)
            m2 = heading_num.match(ln)
            if m1:
                lvl = len(m1.group(1))
                title = m1.group(2).strip()
                current = f"{'#'*lvl} {title}"
                sections.setdefault(current, [])
            elif m2:
                num = m2.group(1)
                title = m2.group(2).strip()
                current = f"## {num}. {title}"
                sections.setdefault(current, [])
            else:
                sections[current].append(ln)

        return {k: "\n".join(v).strip() for k, v in sections.items() if "".join(v).strip()}

    # Key findings 
    _BULLET_RE = re.compile(r"^\s*([\-*•]|\d+[\.)])\s+(.*)$")

    def _extract_key_findings(self, plain_text: str, sections: Dict[str, str]) -> List[str]:
        candidates = [k for k in sections if any(w in k.lower() for w in ("finding", "summary", "conclusion", "results"))]
        bullets: List[str] = []
        for k in candidates or list(sections.keys()):
            for line in sections[k].split("\n"):
                m = self._BULLET_RE.match(line)
                if m:
                    item = m.group(2).strip()
                    if len(item) >= self.cfg.min_finding_len:
                        bullets.append(self._clean_inline(item))
            if len(bullets) >= self.cfg.min_findings:
                break
        if len(bullets) >= self.cfg.min_findings:
            return bullets[: self.cfg.max_findings]

        # Fallback to salient sentences
        sentences = re.split(r"(?<=[.!?])\s+", plain_text)
        scored: List[Tuple[float, str]] = []
        for i, s in enumerate(sentences):
            st = s.strip()
            if len(st) < self.cfg.min_finding_len:
                continue
            length_score = min(len(st) / 200.0, 1.0)
            emphasis = len(re.findall(r"\b[A-Z][a-z]+\b", st)) * 0.05
            position = 1.0 / (1 + i)
            score = length_score * 0.7 + emphasis * 0.2 + position * 0.1    
            scored.append((score, self._clean_inline(st)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[: self.cfg.max_findings]]

    def _clean_inline(self, s: str) -> str:
        # remove inline code backticks while keeping content
        s = re.sub(r"`([^`]+)`", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Tone & category 
    def _guess_tone_category(self, text: str, fm: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
        if "tone" in fm or "category" in fm:
            return fm.get("tone"), fm.get("category")
        tone_map = {
            "urgent": ["critical", "urgent", "severe", "high risk", "immediate"],
            "formal": ["hereby", "therefore", "whereas", "pursuant"],
            "neutral": ["summary", "report", "data", "results"],
            "optimistic": ["improved", "growth", "increase", "opportunity"],
            "cautious": ["limitation", "constraint", "uncertain", "assumption"],
        }
        cat_map = {
            "policy": ["compliance", "policy", "regulation", "gdpr", "hipaa"],
            "engineering": ["architecture", "design", "implementation", "api", "latency"],
            "research": ["method", "dataset", "baseline", "experiment", "significant"],
            "product": ["user", "feature", "roadmap", "feedback", "onboarding"],
            "security": ["threat", "vulnerability", "attack", "mitigation", "encryption"],
        }
        low = text.lower()
        tone = self._keyword_vote(low, tone_map)
        category = self._keyword_vote(low, cat_map)
        return tone, category

    def _keyword_vote(self, low_text: str, mapping: Dict[str, List[str]]) -> Optional[str]:
        best_label, best_count = None, 0
        for label, kws in mapping.items():
            c = sum(low_text.count(k) for k in kws)
            if c > best_count:
                best_label, best_count = label, c
        return best_label if best_count > 0 else None

    # Misc 
    def _rough_token_count(self, s: str) -> int:
        # crude approximation ~4 chars per token
        return int(len(s) / 4)


# CLI

def _cli() -> None:
    p = argparse.ArgumentParser(description="Parse a document and print JSON summary")
    p.add_argument("path", type=str, help="Path to .txt or .md")
    p.add_argument("--json", action="store_true", help="Print full ParsedDocument JSON")
    p.add_argument("--use-llm", action="store_true", help="Use LLM for meta extraction (requires OPENAI_* env)")
    args = p.parse_args()

    agent = ReaderAgent()
    doc = agent.parse(Path(args.path), use_llm=args.use_llm)

    if args.json:
        print(doc.model_dump_json(indent=2, ensure_ascii=False))
    else:
        print(f"Title: {doc.meta.title}")
        if doc.meta.category:
            print(f"Category: {doc.meta.category}")
        if doc.meta.tone:
            print(f"Tone: {doc.meta.tone}")
        print("\nKey findings:")
        for i, k in enumerate(doc.meta.key_findings, 1):
            print(f"  {i}. {k}")
        print("\nSections:")
        for k in list(doc.sections.keys())[:6]:
            preview = doc.sections[k][:120].replace("\n", " ")
            print(f"- {k}: {preview}{'…' if len(doc.sections[k]) > 120 else ''}")

if __name__ == "__main__":  
    _cli()
