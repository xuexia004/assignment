"""Public Crossref metadata search; fulltext links never imply open access."""

from __future__ import annotations

import asyncio
import html
import re
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from paper_agent.config import Settings
from paper_agent.models import ErrorInfo, Paper
from paper_agent.services.retrieval import normalize_doi
from paper_agent.sources.arxiv_source import _RequestGate
from paper_agent.sources.openalex_source import _retry_delay

API_URL = "https://api.crossref.org/works"
# Independent quota; reuse the existing cross-loop-safe gate implementation.
_REQUEST_GATE = _RequestGate()


class CrossrefSourceError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, **details: Any):
        super().__init__(message)
        self.error_info = ErrorInfo(code=code, message=message, stage="crossref_search",
                                    retryable=retryable, details=details)


def _text(value: Any) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value)).split()) if isinstance(value, str) else ""


def _url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        if parts.scheme in {"https", "http"} and parts.hostname and not (parts.username or parts.password):
            return value
    except ValueError:
        pass
    return None


def _publication_date(work: dict) -> tuple[int | None, str | None]:
    for field in ("published", "published-online", "published-print", "issued"):
        entry = work.get(field)
        parts = entry.get("date-parts") if isinstance(entry, dict) else None
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            continue
        values = parts[0]
        if not 1 <= len(values) <= 3 or not all(type(v) is int for v in values):
            continue
        try:
            date(*(values + [1] * (3 - len(values))))
        except (ValueError, TypeError):
            continue
        if values[0] >= 1000:
            return values[0], "-".join([str(values[0]), *(f"{v:02d}" for v in values[1:])])
    return None, None


def parse_work(work: dict[str, Any]) -> Paper:
    doi = normalize_doi(work.get("DOI") if isinstance(work.get("DOI"), str) else None)
    titles = work.get("title")
    title = _text(titles[0]) if isinstance(titles, list) and titles else ""
    if not doi or not title:
        raise CrossrefSourceError("invalid_record", "Crossref 条目缺少有效 DOI 或标题。")
    authors = []
    for author in work.get("author", []) if isinstance(work.get("author"), list) else []:
        if isinstance(author, dict):
            name = " ".join(filter(None, [_text(author.get("given")), _text(author.get("family"))]))
            name = name or _text(author.get("name"))
            if name and name not in authors:
                authors.append(name)
    year, published = _publication_date(work)
    citations = work.get("is-referenced-by-count")
    citations = citations if type(citations) is int and citations >= 0 else None
    links = []
    for link in work.get("link", []) if isinstance(work.get("link"), list) else []:
        if isinstance(link, dict) and (url := _url(link.get("URL"))):
            links.append({"url": url, "content_type": _text(link.get("content-type")),
                          "version": _text(link.get("content-version")),
                          "intended_application": _text(link.get("intended-application"))})
    venues = work.get("container-title")
    retrieved = datetime.now(timezone.utc).isoformat()
    return Paper(
        paper_id=f"crossref:{doi}", title=title, authors=authors, doi=doi, year=year,
        abstract=_text(work.get("abstract")), sources=["crossref"],
        source_records=[{"source": "crossref", "id": doi,
                         "url": f"{API_URL}/{quote(doi, safe='')}", "retrieved_at": retrieved}],
        paper_url=f"https://doi.org/{doi}", citation_count=citations,
        metadata={"publication_date": published, "work_type": _text(work.get("type")),
                  "venue": _text(venues[0]) if isinstance(venues, list) and venues else None,
                  "publisher": _text(work.get("publisher")), "fulltext_links": links,
                  "pdf_urls_verified": False,
                  "citation_metrics": {"crossref": {"count": citations, "retrieved_at": retrieved}}},
    )


async def search_crossref(query: str, limit: int, *, settings: Settings | None = None,
                          client: httpx.AsyncClient | None = None) -> list[Paper]:
    """One bounded page, public pool (one list request/sec, one in flight)."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Crossref query must be non-empty")
    if type(limit) is not int or not 0 <= limit <= 100:
        raise ValueError("Crossref limit must be between 0 and 100")
    if limit == 0:
        return []
    settings = settings or Settings()
    if client is None:
        async with httpx.AsyncClient(proxy=settings.http_proxy or None, trust_env=False,
                                     follow_redirects=False, timeout=settings.http_timeout) as owned:
            return await search_crossref(query, limit, settings=settings, client=owned)
    # query.title is deprecated; bibliographic search is not an exact-title guarantee.
    params = {"query.bibliographic": query.strip(), "rows": limit,
              "filter": "type:journal-article,type:proceedings-article",
              "sort": "relevance", "order": "desc"}
    try:
        async with asyncio.timeout(settings.http_timeout):
            for attempt in range(settings.http_max_retries + 1):
                response = None
                try:
                    async with _REQUEST_GATE.request(1.0):
                        response = await client.get(API_URL, params=params,
                                                    headers={"User-Agent": "PaperAgent/1.0 (Crossref public metadata)"},
                                                    timeout=settings.http_timeout)
                except httpx.RequestError as exc:
                    failure = CrossrefSourceError("network_error", "Crossref 网络请求失败。",
                                                   retryable=True, exception_type=type(exc).__name__)
                else:
                    status = response.status_code
                    if status == 200:
                        break
                    failure = CrossrefSourceError(
                        "rate_limit" if status == 429 else "http_error", f"Crossref 返回 HTTP {status}。",
                        retryable=status == 429 or status >= 500, status_code=status,
                    )
                if not failure.error_info.retryable or attempt >= settings.http_max_retries:
                    raise failure
                await asyncio.sleep(_retry_delay(response, attempt, settings.http_retry_backoff))
    except TimeoutError:
        raise CrossrefSourceError("source_timeout", "Crossref 请求达到总时限。", retryable=True) from None
    try:
        payload = response.json()
        message = payload.get("message") if isinstance(payload, dict) else None
        items = message.get("items") if isinstance(message, dict) else None
    except ValueError:
        items = None
    if not isinstance(items, list):
        raise CrossrefSourceError("invalid_response", "Crossref 未返回有效的论文列表。")
    papers, seen, skipped = [], set(), 0
    for work in items[:limit]:
        try:
            paper = parse_work(work if isinstance(work, dict) else {})
        except CrossrefSourceError:
            skipped += 1
            continue
        if paper.paper_id not in seen:
            seen.add(paper.paper_id)
            papers.append(paper)
    if items and not papers:
        raise CrossrefSourceError("invalid_record", "Crossref 条目均缺少有效 DOI 或标题。", skipped=skipped)
    for paper in papers:
        paper.metadata["source_skipped_records"] = skipped
    return papers
