"""Crossref fixtures verify metadata and transport, never live retrieval quality."""

import asyncio

import httpx
import pytest

from paper_agent.config import Settings
from paper_agent.services.retrieval import merge_dedupe_rank
from paper_agent.sources import crossref_source as source


def work(**values):
    return {"DOI": "10.1000/EXAMPLE", "title": ["Agent memory"],
            "type": "journal-article", **values}


def test_metadata_identity_provenance_dates_and_links_do_not_imply_oa():
    paper = source.parse_work(work(
        abstract="<jats:p>Memory &amp; <jats:italic>retrieval</jats:italic>.</jats:p>",
        author=[{"given": "Alice", "family": "Chen"}, {"name": "Research Team"}, None],
        published={"date-parts": [[2025, 2]]},
        **{"is-referenced-by-count": 20, "container-title": ["Journal A"],
           "link": [{"URL": "https://publisher.example/paper.pdf", "content-type": "application/pdf"},
                    {"URL": "file:///secret.pdf"}]},
    ))
    assert paper.doi == "10.1000/example" and paper.paper_id == "crossref:10.1000/example"
    assert paper.authors == ["Alice Chen", "Research Team"]
    assert paper.abstract == "Memory & retrieval ."
    assert paper.year == 2025 and paper.metadata["publication_date"] == "2025-02"
    assert paper.citation_count == 20
    assert paper.metadata["citation_metrics"]["crossref"]["count"] == 20
    assert len(paper.metadata["fulltext_links"]) == 1
    assert paper.source_records[0]["retrieved_at"]
    assert paper.pdf_url is None and paper.pdf_urls == [] and paper.pdf_path is None


@pytest.mark.parametrize("values", [{"DOI": "bad"}, {"title": []}, {"DOI": None}])
def test_invalid_id_or_title_is_explicit(values):
    with pytest.raises(source.CrossrefSourceError):
        source.parse_work(work(**values))


def test_invalid_dates_and_citation_values_stay_unknown():
    paper = source.parse_work(work(published={"date-parts": [[2025, 2, 31]]},
                                   **{"is-referenced-by-count": True}))
    assert paper.year is None and paper.citation_count is None


def test_cross_source_doi_merge_preserves_original_and_crossref_metadata():
    from paper_agent.models import Paper
    original = Paper(paper_id="arxiv:1234", title="Agent memory", doi="10.1000/example",
                     sources=["arxiv"], abstract="Longer original abstract", pdf_url="https://arxiv.org/pdf/1234")
    merged = merge_dedupe_rank([original, source.parse_work(work())], "agent memory")
    assert len(merged) == 1 and set(merged[0].sources) == {"arxiv", "crossref"}
    assert merged[0].abstract == original.abstract and merged[0].pdf_url == original.pdf_url
    assert original.sources == ["arxiv"]


def search_with_responses(monkeypatch, handler, **options):
    class NoWaitGate:
        def request(self, interval):
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def gate():
                assert interval == 1.0
                yield
            return gate()
    monkeypatch.setattr(source, "_REQUEST_GATE", NoWaitGate())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await source.search_crossref("agent memory", 3, client=client,
                                                   settings=Settings(http_retry_backoff=0, **options))
            assert not client.is_closed
            return result
    return asyncio.run(run())


def test_public_request_bounded_list_and_duplicate_invalid_records(monkeypatch):
    def handle(request):
        assert request.url.params["rows"] == "3"
        assert request.url.params["query.bibliographic"] == "agent memory"
        assert "authorization" not in request.headers and "api_key" not in request.url.params
        return httpx.Response(200, json={"message": {"items": [work(), work(), {"title": []}]}})
    papers = search_with_responses(monkeypatch, handle)
    assert len(papers) == 1 and papers[0].metadata["source_skipped_records"] == 1


def test_rate_limit_retry_then_success(monkeypatch):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "0"}) if len(requests) == 1 else httpx.Response(
            200, json={"message": {"items": [work()]}})
    assert len(search_with_responses(monkeypatch, handle)) == 1
    assert len(requests) == 2


@pytest.mark.parametrize("status", [403, 429, 500])
def test_http_failures_are_not_empty_success(monkeypatch, status):
    with pytest.raises(source.CrossrefSourceError) as error:
        search_with_responses(monkeypatch, lambda _: httpx.Response(status), http_max_retries=0)
    assert error.value.error_info.details["status_code"] == status


@pytest.mark.parametrize("payload", [{}, {"message": {"items": [{}]}}, {"message": {"items": "bad"}}])
def test_malformed_payload_is_explicit(monkeypatch, payload):
    with pytest.raises(source.CrossrefSourceError):
        search_with_responses(monkeypatch, lambda _: httpx.Response(200, json=payload))


def test_genuine_empty_search_is_valid(monkeypatch):
    assert search_with_responses(monkeypatch, lambda _: httpx.Response(200, json={"message": {"items": []}})) == []


def test_stage_timeout_includes_network_wait(monkeypatch):
    async def handle(request):
        await asyncio.Event().wait()
    with pytest.raises(source.CrossrefSourceError) as error:
        search_with_responses(monkeypatch, handle, http_timeout=.01)
    assert error.value.error_info.code == "source_timeout"
