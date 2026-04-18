from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICE_PATH = PROJECT_ROOT / "artifacts" / "input" / "price.json"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "", str(model_name or "")).strip().lower()
    return slug or "model"


def tokenize_text(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(str(text or ""))]


def count_text_tokens(text: str) -> int:
    return len(tokenize_text(text))


def parse_literal(value: object, *, default: Any) -> Any:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default
        try:
            return ast.literal_eval(raw)
        except Exception:
            return default
    return default


def parse_literal_list(value: object) -> List[Any]:
    parsed = parse_literal(value, default=[])
    if isinstance(parsed, list):
        return list(parsed)
    return []


def normalize_domain(name: str) -> str:
    text = str(name or "").strip().lower()
    if text == "crm":
        return "customer_relationship_manager"
    return text


def normalize_domains(raw: object) -> List[str]:
    values = parse_literal_list(raw)
    return [normalize_domain(item) for item in values if str(item or "").strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_price_table() -> Dict[str, Dict[str, float]]:
    payload = json.loads(PRICE_PATH.read_text(encoding="utf-8"))
    return {str(key): {str(k): float(v) for k, v in value.items()} for key, value in payload.items()}


def resolve_price_key(model_name: str, price_table: Optional[Mapping[str, object]] = None) -> str:
    table = price_table or load_price_table()
    if model_name in table:
        return str(model_name)

    model_norm = str(model_name or "").strip().lower()
    for key in table.keys():
        if str(key).lower() in model_norm:
            return str(key)

    if "deepseek" in model_norm and "deepseek-chat" in table:
        return "deepseek-chat"

    raise ValueError(f"Could not map model `{model_name}` to a price key.")


@dataclass
class TokenUsage:
    calls: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    approx_calls: int = 0

    @classmethod
    def from_counter(cls, counter: object) -> "TokenUsage":
        return cls(
            calls=int(getattr(counter, "calls", 0) or 0),
            prompt_cache_hit_tokens=int(getattr(counter, "prompt_cache_hit_tokens", 0) or 0),
            prompt_cache_miss_tokens=int(getattr(counter, "prompt_cache_miss_tokens", 0) or 0),
            completion_tokens=int(getattr(counter, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(counter, "total_tokens", 0) or 0),
            prompt_tokens=int(getattr(counter, "prompt_tokens", 0) or 0),
            approx_calls=int(getattr(counter, "approx_calls", 0) or 0),
        )

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, object]], *, calls: int = 0) -> "TokenUsage":
        if payload is None:
            return cls(calls=calls)
        prompt_tokens = int(payload.get("prompt_tokens") or 0)
        cache_hit = int(payload.get("prompt_cache_hit_tokens") or 0)
        cache_miss = payload.get("prompt_cache_miss_tokens")
        if cache_miss is None:
            cache_miss = max(0, prompt_tokens - cache_hit)
        return cls(
            calls=int(calls or 0),
            prompt_cache_hit_tokens=cache_hit,
            prompt_cache_miss_tokens=int(cache_miss or 0),
            completion_tokens=int(payload.get("completion_tokens") or payload.get("output_tokens") or 0),
            total_tokens=int(payload.get("total_tokens") or 0),
            prompt_tokens=prompt_tokens,
            approx_calls=0,
        )

    def add(self, other: "TokenUsage") -> None:
        self.calls += int(other.calls)
        self.prompt_cache_hit_tokens += int(other.prompt_cache_hit_tokens)
        self.prompt_cache_miss_tokens += int(other.prompt_cache_miss_tokens)
        self.completion_tokens += int(other.completion_tokens)
        self.total_tokens += int(other.total_tokens)
        self.prompt_tokens += int(other.prompt_tokens)
        self.approx_calls += int(other.approx_calls)

    def merged(self, other: "TokenUsage") -> "TokenUsage":
        merged = TokenUsage(**self.to_dict())
        merged.add(other)
        return merged

    def to_dict(self) -> Dict[str, int]:
        return {
            "calls": int(self.calls),
            "prompt_cache_hit_tokens": int(self.prompt_cache_hit_tokens),
            "prompt_cache_miss_tokens": int(self.prompt_cache_miss_tokens),
            "completion_tokens": int(self.completion_tokens),
            "output_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
            "prompt_tokens": int(self.prompt_tokens),
            "approx_calls": int(self.approx_calls),
        }

    def to_cost_line(self) -> str:
        return (
            f"calls={int(self.calls)} "
            f"prompt_cache_hit={int(self.prompt_cache_hit_tokens)} "
            f"prompt_cache_miss={int(self.prompt_cache_miss_tokens)} "
            f"output={int(self.completion_tokens)} "
            f"total={int(self.total_tokens)} "
            f"approx_calls={int(self.approx_calls)}"
        )


def aggregate_token_usage(usages: Iterable[TokenUsage]) -> TokenUsage:
    total = TokenUsage()
    for usage in usages:
        total.add(usage)
    return total


def compute_price_usd(*, model_name: str, usage: TokenUsage, price_table: Optional[Mapping[str, object]] = None) -> float:
    table = price_table or load_price_table()
    key = resolve_price_key(model_name, table)
    rates = table[key]

    getcontext().prec = 28
    million = Decimal(1_000_000)

    hit = Decimal(int(usage.prompt_cache_hit_tokens))
    miss = Decimal(int(usage.prompt_cache_miss_tokens))
    output = Decimal(int(usage.completion_tokens))

    total = (
        (hit / million) * Decimal(str(rates["prompt_cache_hit"]))
        + (miss / million) * Decimal(str(rates["prompt_cache_miss"]))
        + (output / million) * Decimal(str(rates["output"]))
    )
    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def sum_mapping_values(values: Iterable[float]) -> float:
    total = Decimal("0")
    for value in values:
        total += Decimal(str(value))
    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class RetrievalDocument:
    doc_id: str
    query: str
    search_text: str
    rendered_text: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    score: float
    query: str
    rendered_text: str
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "score": float(self.score),
            "query": self.query,
            "rendered_text": self.rendered_text,
            "metadata": dict(self.metadata),
        }


class LexicalRetriever:
    def __init__(self, documents: Sequence[RetrievalDocument]) -> None:
        self.documents = list(documents)
        self._docs_by_id = {doc.doc_id: doc for doc in self.documents}
        self._term_freqs: Dict[str, Counter[str]] = {}
        self._doc_lengths: Dict[str, int] = {}
        doc_freq: Counter[str] = Counter()
        total_length = 0

        for doc in self.documents:
            tokens = tokenize_text(doc.search_text)
            counts = Counter(tokens)
            self._term_freqs[doc.doc_id] = counts
            self._doc_lengths[doc.doc_id] = len(tokens)
            total_length += len(tokens)
            for token in counts:
                doc_freq[token] += 1

        self._avg_doc_len = (total_length / len(self.documents)) if self.documents else 1.0
        self._idf: Dict[str, float] = {}
        n_docs = max(1, len(self.documents))
        for token, freq in doc_freq.items():
            self._idf[token] = math.log(1.0 + ((n_docs - freq + 0.5) / (freq + 0.5)))

    def search(
        self,
        query: str,
        *,
        top_k: int,
        candidate_doc_ids: Optional[Sequence[str]] = None,
        exclude_exact_query: bool = True,
    ) -> List[RetrievalHit]:
        query_terms = tokenize_text(query)
        if not query_terms:
            return []

        candidates: Iterable[RetrievalDocument]
        if candidate_doc_ids is None:
            candidates = self.documents
        else:
            allowed = {str(doc_id) for doc_id in candidate_doc_ids}
            candidates = [self._docs_by_id[doc_id] for doc_id in allowed if doc_id in self._docs_by_id]

        k1 = 1.5
        b = 0.75
        exact_query = str(query or "").strip().lower()
        scored: List[Tuple[float, str]] = []

        for doc in candidates:
            if exclude_exact_query and str(doc.query or "").strip().lower() == exact_query:
                continue
            counts = self._term_freqs.get(doc.doc_id, Counter())
            doc_len = self._doc_lengths.get(doc.doc_id, 0)
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + k1 * (1.0 - b + b * (doc_len / max(self._avg_doc_len, 1.0)))
                score += idf * (tf * (k1 + 1.0) / max(denom, 1e-9))
            scored.append((score, doc.doc_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        hits: List[RetrievalHit] = []
        for score, doc_id in scored[: max(int(top_k), 0)]:
            doc = self._docs_by_id[doc_id]
            hits.append(
                RetrievalHit(
                    doc_id=doc.doc_id,
                    score=float(score),
                    query=doc.query,
                    rendered_text=doc.rendered_text,
                    metadata=dict(doc.metadata),
                )
            )
        return hits


def render_retrieval_context(
    hits: Sequence[RetrievalHit],
    *,
    max_context_tokens: int,
    intro: str,
) -> Tuple[str, List[RetrievalHit], int]:
    blocks: List[str] = []
    kept: List[RetrievalHit] = []
    used_tokens = 0

    for rank, hit in enumerate(hits, start=1):
        block = f"Example {rank} (score={hit.score:.4f})\n{hit.rendered_text.strip()}"
        block_tokens = count_text_tokens(block)
        if blocks and used_tokens + block_tokens > max_context_tokens:
            break
        blocks.append(block)
        kept.append(hit)
        used_tokens += block_tokens

    if not blocks:
        return "", [], 0
    text = f"{intro.strip()}\n\n" + "\n\n".join(blocks)
    return text, kept, used_tokens
