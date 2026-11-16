"""Downloader tests use generated PDFs and explicit mock DNS; no live network."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import httpx
import pymupdf
import pytest

from paper_agent.config import Settings
from paper_agent.models import PageMapping, Paper
from paper_agent.services.downloader import Downloader, download_pdf


def pdf_bytes(text="Mock test fixture: a readable PDF, not a research paper."):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), text)
        return document.tobytes()


async def public_dns(host, port):
    return ["93.184.216.34"]


def configuration(tmp_path, **kwargs):
    return Settings(data_dir=tmp_path, http_max_retries=1, http_retry_backoff=0, **kwargs)


def paper(**kwargs):
    return Paper(paper_id="../../unsafe:paper/1", title="Mock fixture", **kwargs)


def run_download(tmp_path, handler, document=None, **settings):
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await Downloader(
                settings=configuration(tmp_path, **settings), client=client, resolver=public_dns
            ).download(document or paper(pdf_url="https://papers.example/one.pdf"))
            assert not client.is_closed
            return result

    return asyncio.run(execute())


def test_success_copy_atomic_safe_filename_and_tls_host_pinning(tmp_path):
    original = paper(pdf_url="https://papers.example/one.pdf?token=not-logged")
    payload = pdf_bytes()

    def handler(request):
        assert request.url.host == "93.184.216.34"
        assert request.headers["Host"] == "papers.example"
        assert request.extensions["sni_hostname"] == "papers.example"
        assert request.url.params["token"] == "not-logged"
        return httpx.Response(200, content=payload)

    result = run_download(tmp_path, handler, original)
    assert result.status == "downloaded"
    assert result.acquisition_status == "downloaded"
    assert result.conversion_status == "not_attempted"
    assert result.reading_status == "metadata_only"
    assert result.pdf_artifact_id == hashlib.sha256(
        f"pdf:{original.paper_id}".encode("utf-8")
    ).hexdigest()[:20]
    assert result.markdown_artifact_id is None
    assert original.pdf_path is None and original.status == "metadata_only"
    assert original.metadata == {}
    path = Path(result.pdf_path)
    assert path.parent == tmp_path / "pdfs"
    assert path.name == hashlib.sha256(original.paper_id.encode()).hexdigest() + ".pdf"
    assert path.read_bytes() == payload
    assert not list(path.parent.glob("*.tmp"))
    assert result.metadata["downloads"]["request_count"] == 1
    assert result.metadata["downloads"]["pages"] == 1
    assert "token" not in json.dumps(result.metadata["downloads"])


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_unavailable_status_is_explicit(tmp_path, status):
    result = run_download(tmp_path, lambda request: httpx.Response(status))
    assert result.status == "no_open_access" and result.pdf_path is None
    assert result.acquisition_status == "unavailable"
    assert result.conversion_status == "not_attempted"
    assert result.reading_status == "metadata_only"
    assert result.pdf_artifact_id is None
    assert result.errors[-1].code == "access_unavailable"
    assert result.errors[-1].details["status_code"] == status
    assert result.metadata["downloads"]["request_count"] == 1
    assert not list(tmp_path.rglob("*.pdf"))


def test_failed_refresh_preserves_existing_pdf_artifact(tmp_path):
    path = tmp_path / "existing.pdf"
    payload = pdf_bytes()
    path.write_bytes(payload)
    original = paper(
        pdf_url="https://papers.example/removed.pdf",
        pdf_path=str(path),
        acquisition_status="downloaded",
    )

    result = run_download(tmp_path, lambda request: httpx.Response(404), original)

    assert result.status == "no_open_access"
    assert result.acquisition_status == "unavailable"
    assert result.pdf_path == str(path)
    assert result.pdf_artifact_id == hashlib.sha256(
        f"pdf:{original.paper_id}".encode("utf-8")
    ).hexdigest()[:20]
    assert path.read_bytes() == payload


@pytest.mark.parametrize("content", [b"<html>Sign in</html>", b"%PDF-1.7\nthis is not a PDF"])
def test_fake_and_broken_pdf_never_saved(tmp_path, content):
    result = run_download(tmp_path, lambda request: httpx.Response(200, content=content))
    assert result.status == "download_failed"
    assert result.errors[-1].code == "invalid_pdf"
    assert not list(tmp_path.rglob("*.pdf"))


def test_fallback_preserves_errors_and_final_candidate(tmp_path):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == "/bad.pdf":
            return httpx.Response(404)
        return httpx.Response(200, content=pdf_bytes())

    result = run_download(tmp_path, handler, paper(
        pdf_url="https://papers.example/bad.pdf",
        pdf_urls=["https://papers.example/bad.pdf", "https://papers.example/good.pdf"],
    ))
    assert seen == ["/bad.pdf", "/good.pdf"]
    assert result.status == "downloaded" and result.error is None
    assert result.errors[0].code == "access_unavailable"
    assert result.metadata["downloads"]["candidate"].endswith("/good.pdf")
    assert result.metadata["downloads"]["request_count"] == 2


def test_timeout_retries_are_bounded(tmp_path):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        raise httpx.ReadTimeout("sensitive URL must not be logged", request=request)

    result = run_download(tmp_path, handler)
    assert result.status == "download_failed"
    assert result.acquisition_status == "failed"
    assert result.conversion_status == "not_attempted"
    assert count == 2 and result.metadata["downloads"]["retries"] == 1
    assert result.errors[-1].code == "timeout"
    assert "sensitive" not in result.model_dump_json()


def test_retryable_http_then_success(tmp_path):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(503) if count == 1 else httpx.Response(200, content=pdf_bytes())

    result = run_download(tmp_path, handler)
    assert result.status == "downloaded"
    assert result.metadata["downloads"]["request_count"] == 2


def test_candidate_timeout_falls_back_to_next_candidate(tmp_path):
    async def handler(request):
        if request.url.path == "/slow.pdf":
            await asyncio.sleep(0.2)
            return httpx.Response(200, content=pdf_bytes())
        return httpx.Response(200, content=pdf_bytes())

    document = paper(pdf_url="https://papers.example/slow.pdf",
                     pdf_urls=["https://papers.example/slow.pdf", "https://papers.example/good.pdf"])
    result = run_download(tmp_path, handler, document, download_timeout=1,
                          download_candidate_timeout=0.05, http_timeout=0.2)
    assert result.status == "downloaded"
    assert result.metadata["downloads"]["request_count"] == 2
    assert any(error["code"] == "candidate_timeout" for error in result.metadata["downloads"]["errors"])


def test_total_timeout_covers_all_attempts(tmp_path):
    async def handler(request):
        await asyncio.sleep(0.2)
        return httpx.Response(200, content=pdf_bytes())

    result = run_download(tmp_path, handler, download_timeout=0.05)
    assert result.status == "download_failed"
    assert result.errors[-1].code == "download_timeout"


@pytest.mark.parametrize("total_budget,success", [(1.0, True), (0.05, False)])
def test_source_queue_does_not_consume_candidate_budget_but_total_still_applies(tmp_path, monkeypatch, total_budget, success):
    @asynccontextmanager
    async def delayed_gate(host, interval):
        await asyncio.sleep(0.12)
        yield

    monkeypatch.setattr("paper_agent.services.downloader._source_gate", delayed_gate)
    payload = pdf_bytes()
    result = run_download(tmp_path, lambda request: httpx.Response(200, content=payload),
                          download_timeout=total_budget, download_candidate_timeout=0.04)
    if success:
        assert result.status == "downloaded"
        assert result.metadata["downloads"]["queue_seconds"] >= 0.1
        assert result.metadata["downloads"]["received_bytes"] == len(payload)
    else:
        assert result.errors[-1].code == "download_timeout"
        assert result.metadata["downloads"]["request_count"] == 0
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("declared", [False, True])
def test_byte_limit_streamed_and_content_length(tmp_path, declared):
    body = pdf_bytes()
    headers = {"Content-Length": str(len(body))} if declared else {"Content-Length": "unknown"}
    result = run_download(
        tmp_path, lambda request: httpx.Response(200, content=body, headers=headers),
        max_pdf_bytes=10,
    )
    assert result.status == "download_failed" and result.errors[-1].code == "pdf_too_large"
    assert not list(tmp_path.rglob("*.pdf"))


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://papers.example/a.pdf", "http://localhost/a.pdf",
    "https://user:password@papers.example/a.pdf", "http://127.0.0.1/a.pdf",
    "http://10.1.2.3/a.pdf", "http://169.254.169.254/latest/meta-data",
    "http://[::1]/a.pdf", "http://[::ffff:127.0.0.1]/a.pdf", "http://service.internal/a.pdf",
    "http://papers.example/a\r\nHost:localhost", "https://[fe80::1%25eth0]/a.pdf",
])
def test_unsafe_candidates_never_reach_transport(tmp_path, url):
    def handler(request):
        pytest.fail("Unsafe URL was requested")

    result = run_download(tmp_path, handler, paper(pdf_url=url))
    assert result.status == "download_failed"
    assert result.errors[-1].code == "unsafe_url"
    assert result.metadata["downloads"]["request_count"] == 0


def test_dns_private_or_mixed_answer_is_rejected_even_with_mock_client(tmp_path):
    async def execute():
        async def mixed_dns(host, port):
            return ["93.184.216.34", "10.0.0.1"]

        def handler(request):
            pytest.fail("DNS validation must not be bypassed for MockTransport")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await Downloader(
                settings=configuration(tmp_path), client=client, resolver=mixed_dns
            ).download(paper(pdf_url="https://papers.example/a.pdf"))

    result = asyncio.run(execute())
    assert result.status == "download_failed"
    assert result.errors[-1].code == "unsafe_url"


def test_redirect_revalidates_dns_and_blocks_private_destination(tmp_path):
    async def execute():
        count = 0

        async def resolver(host, port):
            return ["127.0.0.1"] if host == "internal.example" else ["93.184.216.34"]

        def handler(request):
            nonlocal count
            count += 1
            return httpx.Response(302, headers={"Location": "https://internal.example/private.pdf"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await Downloader(
                settings=configuration(tmp_path), client=client, resolver=resolver
            ).download(paper(pdf_url="https://papers.example/a.pdf"))
        assert count == 1
        return result

    result = asyncio.run(execute())
    assert result.errors[-1].code == "unsafe_url"
    assert result.metadata["downloads"]["request_count"] == 1


def test_relative_redirect_and_redirect_limit(tmp_path):
    def handler(request):
        if request.url.path == "/one.pdf":
            return httpx.Response(302, headers={"Location": "/final.pdf"})
        return httpx.Response(200, content=pdf_bytes())

    result = run_download(tmp_path, handler)
    assert result.status == "downloaded" and result.metadata["downloads"]["redirects"] == 1
    other = run_download(tmp_path / "limited", handler, max_redirects=0)
    assert other.errors[-1].code == "too_many_redirects"
    assert other.metadata["downloads"]["request_count"] == 1


def test_cache_is_hashed_and_readability_verified(tmp_path):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(200, content=pdf_bytes())

    first = run_download(tmp_path, handler)
    second = run_download(tmp_path, handler)
    assert count == 1 and second.metadata["downloads"]["cache_hit"] is True
    assert second.metadata["downloads"]["request_count"] == 0
    Path(first.pdf_path).write_bytes(b"%PDF-1.7\ncorrupt")
    third = run_download(tmp_path, handler)
    assert count == 2 and third.metadata["downloads"]["cache_hit"] is False
    assert third.status == "downloaded"
    manifest = Path(first.pdf_path).with_suffix(".json")
    content = json.loads(manifest.read_text(encoding="utf-8"))
    content["sha256"] = "0" * 64
    manifest.write_text(json.dumps(content), encoding="utf-8")
    fourth = run_download(tmp_path, handler)
    assert count == 3 and fourth.status == "downloaded"


@pytest.mark.parametrize("infer_from_url", [False, True])
def test_arxiv_revisions_have_separate_verified_caches(tmp_path, monkeypatch, infer_from_url):
    @asynccontextmanager
    async def no_delay(host, interval):
        yield

    # Only replace rate-limit sleeps. Mock DNS and the normal safe fetch remain active.
    monkeypatch.setattr("paper_agent.services.downloader._source_gate", no_delay)
    payloads = {"v1": pdf_bytes("Version one fixture."), "v2": pdf_bytes("Version two fixture.")}
    requests = []

    def document(version):
        host = "arxiv.org" if infer_from_url else "papers.example"
        return Paper(paper_id="arxiv:2601.01234", title="Revision fixture", arxiv_id="2601.01234",
                     arxiv_version=None if infer_from_url else version,
                     pdf_url=f"https://{host}/pdf/2601.01234{version}")

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, content=payloads[request.url.path[-2:]])

    first = run_download(tmp_path, handler, document("v1"))
    second = run_download(tmp_path, handler, document("v2"))
    reread = run_download(tmp_path, handler, document("v1"))
    assert len(requests) == 2
    assert first.pdf_path != second.pdf_path
    assert first.metadata["downloads"]["sha256"] != second.metadata["downloads"]["sha256"]
    assert Path(first.pdf_path).read_bytes() == payloads["v1"]
    assert Path(second.pdf_path).read_bytes() == payloads["v2"]
    assert reread.pdf_path == first.pdf_path and reread.metadata["downloads"]["cache_hit"]
    manifest_path = Path(first.pdf_path).with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["arxiv_id"] == "2601.01234" and manifest["arxiv_version"] == "v1"
    manifest["arxiv_version"] = "v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    repaired = run_download(tmp_path, handler, document("v1"))
    assert len(requests) == 3 and not repaired.metadata["downloads"]["cache_hit"]
    assert Path(second.pdf_path).read_bytes() == payloads["v2"]


def test_legacy_cache_remains_unversioned_and_cannot_satisfy_explicit_revision(tmp_path):
    document = Paper(paper_id="arxiv:2601.01234", title="Legacy fixture", arxiv_id="2601.01234",
                     pdf_url="https://papers.example/latest.pdf")
    legacy = pdf_bytes("Legacy cached fixture.")
    current = pdf_bytes("Version two fixture.")
    key = hashlib.sha256(document.paper_id.encode()).hexdigest()
    path = tmp_path / "pdfs" / f"{key}.pdf"
    path.parent.mkdir()
    path.write_bytes(legacy)
    path.with_suffix(".json").write_text(json.dumps({
        "schema": 1, "paper_id": document.paper_id, "candidate": document.pdf_url,
        "sha256": hashlib.sha256(legacy).hexdigest(), "bytes": len(legacy), "pages": 1,
    }), encoding="utf-8")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=current)

    old = run_download(tmp_path, handler, document)
    new = run_download(tmp_path, handler, document.model_copy(update={"arxiv_version": "v2"}))
    reread = run_download(tmp_path, handler, document)
    assert len(requests) == 1
    assert old.metadata["downloads"]["cache_hit"] and reread.metadata["downloads"]["cache_hit"]
    assert not new.metadata["downloads"]["cache_hit"] and new.pdf_path != str(path)
    assert path.read_bytes() == legacy and Path(new.pdf_path).read_bytes() == current


@pytest.mark.parametrize("new_status", [200, 404])
@pytest.mark.parametrize("keyword_source", ["pdf_keyword_section", "author_metadata"])
def test_revision_change_detaches_inherited_fulltext_without_deleting_files(tmp_path, new_status, keyword_source):
    original = Paper(paper_id="arxiv:2601.01234", title="Revision fixture", arxiv_id="2601.01234",
                     arxiv_version="v1", pdf_url="https://papers.example/v1.pdf")
    previous_bytes = pdf_bytes("Version one fixture.")
    previous = run_download(tmp_path, lambda request: httpx.Response(200, content=previous_bytes), original)
    markdown = tmp_path / "old.md"
    markdown.write_text("Version one extracted text.", encoding="utf-8")
    previous.markdown_path = str(markdown)
    previous.markdown_artifact_id = "old-markdown"
    previous.page_map = [PageMapping(page_number=1, text="Version one extracted text.")]
    previous.fulltext_quality = {"complete": True}
    previous.conversion_status = "converted"
    previous.metadata["conversion"] = {"parser": "old-parser"}
    previous.metadata["research_audit"] = {"source_scope": "fulltext"}
    previous.metadata["author_keywords"] = ["Previous keyword"]
    previous.metadata["author_keywords_provenance"] = {"source": keyword_source}
    requested = previous.model_copy(deep=True, update={
        "arxiv_version": "v2", "pdf_url": "https://papers.example/v2.pdf",
    })
    result = run_download(tmp_path, lambda request: httpx.Response(new_status, content=pdf_bytes()), requested)
    assert result.markdown_path is None and result.markdown_artifact_id is None
    assert result.page_map == [] and result.fulltext_quality == {}
    assert result.conversion_status == "not_attempted" and result.reading_status == "metadata_only"
    assert "conversion" not in result.metadata and "research_audit" not in result.metadata
    assert result.metadata["downloads"]["previous_revision_detached"]["arxiv_version"] == "v1"
    if keyword_source == "pdf_keyword_section":
        assert "author_keywords" not in result.metadata and "author_keywords_provenance" not in result.metadata
    else:
        assert result.metadata["author_keywords"] == ["Previous keyword"]
    if new_status == 404:
        assert result.pdf_path is None and result.pdf_artifact_id is None
    else:
        assert result.pdf_path != previous.pdf_path
    assert requested.pdf_path == previous.pdf_path and requested.markdown_path == str(markdown)
    assert requested.metadata["author_keywords"] == ["Previous keyword"]
    assert Path(previous.pdf_path).read_bytes() == previous_bytes
    assert markdown.read_text(encoding="utf-8") == "Version one extracted text."


@pytest.mark.parametrize("first_suffix", ["", "v1"])
def test_explicit_revision_pins_latest_url_and_rejects_wrong_version(tmp_path, monkeypatch, first_suffix):
    @asynccontextmanager
    async def no_delay(host, interval):
        yield

    monkeypatch.setattr("paper_agent.services.downloader._source_gate", no_delay)
    requests = []

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, content=pdf_bytes())

    document = Paper(paper_id="arxiv:2601.01234", title="Revision fixture",
                     arxiv_id="2601.01234", arxiv_version="v2",
                     pdf_url=f"https://arxiv.org/pdf/2601.01234{first_suffix}",
                     pdf_urls=["https://arxiv.org/pdf/2601.01234v2"])
    result = run_download(tmp_path, handler, document)
    assert result.status == "downloaded"
    assert requests == ["/pdf/2601.01234v2"]
    assert result.metadata["downloads"]["cache_identity"]["arxiv_version"] == "v2"
    assert bool(result.errors) is bool(first_suffix)
    if first_suffix:
        assert result.errors[0].code == "arxiv_version_mismatch"


def test_ambiguous_arxiv_revision_candidates_do_not_create_or_reuse_cache(tmp_path, monkeypatch):
    @asynccontextmanager
    async def no_delay(host, interval):
        yield

    monkeypatch.setattr("paper_agent.services.downloader._source_gate", no_delay)
    requests = []

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, content=pdf_bytes())

    document = Paper(paper_id="arxiv:2601.01234", title="Ambiguous fixture", arxiv_id="2601.01234",
                     pdf_urls=["https://arxiv.org/pdf/2601.01234v1", "https://arxiv.org/pdf/2601.01234v2"])
    first = run_download(tmp_path, handler, document)
    second = run_download(tmp_path, handler, document)
    assert len(requests) == 2
    assert first.metadata["downloads"]["ambiguous_arxiv_version"]
    assert not second.metadata["downloads"]["cache_hit"]
    assert not list((tmp_path / "pdfs").glob("*.json"))


@pytest.mark.parametrize("redirect_path", ["/pdf/2601.01234v1", "/pdf/2601.99999v2"])
def test_redirect_cannot_cache_a_different_official_arxiv_revision(tmp_path, monkeypatch, redirect_path):
    @asynccontextmanager
    async def no_delay(host, interval):
        yield

    monkeypatch.setattr("paper_agent.services.downloader._source_gate", no_delay)
    requests = []

    def handler(request):
        requests.append(request.url.path)
        if len(requests) == 1:
            return httpx.Response(302, headers={"Location": "https://arxiv.org" + redirect_path})
        pytest.fail("A mismatched revision must be rejected before sending the redirected request")

    document = Paper(paper_id="arxiv:2601.01234", title="Redirect fixture", arxiv_id="2601.01234",
                     arxiv_version="v2", pdf_url="https://arxiv.org/pdf/2601.01234v2")
    result = run_download(tmp_path, handler, document)
    assert result.status == "download_failed" and result.pdf_path is None
    assert requests == ["/pdf/2601.01234v2"]
    assert result.errors[-1].code == "arxiv_version_mismatch"
    assert result.metadata["downloads"]["redirects"] == 1
    assert not list((tmp_path / "pdfs").glob("*"))


@pytest.mark.parametrize("mode", ["off", "read_only"])
def test_nonwrite_cache_modes_still_write_artifact_without_manifest(tmp_path, mode):
    result = run_download(
        tmp_path, lambda request: httpx.Response(200, content=pdf_bytes()), cache_mode=mode
    )
    assert result.status == "downloaded" and Path(result.pdf_path).exists()
    assert not list(tmp_path.rglob("*.json"))


def test_module_entry_no_candidate_and_original_unchanged(tmp_path):
    original = paper()
    result = asyncio.run(download_pdf(original, settings=configuration(tmp_path)))
    assert result.status == "no_open_access"
    assert result.acquisition_status == "unavailable"
    assert result.conversion_status == "not_attempted"
    assert result.reading_status == "metadata_only"
    assert result.errors[-1].code == "no_open_access"
    assert original.status == "metadata_only" and not original.errors


def test_unavailable_pdf_does_not_hide_abstract_reading_level(tmp_path):
    result = asyncio.run(download_pdf(
        paper(abstract="A real abstract remains available for limited analysis."),
        settings=configuration(tmp_path),
    ))
    assert result.acquisition_status == "unavailable"
    assert result.reading_status == "abstract_only"


def test_dns_failure_becomes_per_paper_error(tmp_path):
    async def execute():
        async def broken_dns(host, port):
            raise OSError("DNS unavailable")

        return await Downloader(
            settings=configuration(tmp_path), resolver=broken_dns
        ).download(paper(pdf_url="https://papers.example/a.pdf"))

    result = asyncio.run(execute())
    assert result.status == "download_failed" and result.errors[-1].code == "dns_error"


def test_search_and_download_share_the_same_arxiv_gate():
    from paper_agent.services import downloader
    from paper_agent.sources import arxiv_source

    assert downloader._REQUEST_GATE is arxiv_source._REQUEST_GATE


def test_cache_read_only_preserves_existing_cache_after_hit(tmp_path):
    def handler(request):
        return httpx.Response(200, content=pdf_bytes())

    first = run_download(tmp_path, handler)
    path = Path(first.pdf_path)
    before = path.stat().st_mtime_ns
    second = run_download(tmp_path, handler, cache_mode="read_only")
    assert second.metadata["downloads"]["cache_hit"]
    assert path.stat().st_mtime_ns == before


def test_public_literal_does_not_call_dns(tmp_path):
    async def execute():
        async def forbidden_dns(host, port):
            pytest.fail("Literal addresses do not need DNS")

        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=pdf_bytes())
        )) as client:
            return await Downloader(
                settings=replace(configuration(tmp_path), cache_enabled=False),
                client=client, resolver=forbidden_dns,
            ).download(paper(pdf_url="https://93.184.216.34/a.pdf"))

    assert asyncio.run(execute()).status == "downloaded"


@pytest.mark.parametrize("proxy", ["http://proxy.example:8080", "https://proxy.example:8080"])
def test_real_httpcore_proxy_tunnel_keeps_original_tls_hostname(tmp_path, proxy):
    """Exercise HTTPX/httpcore proxy code with fake sockets, no network or TLS bypass."""
    import httpcore

    payload = pdf_bytes()
    tls_names, writes = [], []

    class Socket(httpcore.AsyncNetworkStream):
        def __init__(self):
            self.phase = "connect"
            self.read_phases = set()

        async def read(self, max_bytes, timeout=None):
            if self.phase in self.read_phases:
                return b""
            self.read_phases.add(self.phase)
            if self.phase == "connect":
                return b"HTTP/1.1 200 Connection established\r\n\r\n"
            return (
                b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(payload)).encode()
                + b"\r\nConnection: close\r\n\r\n" + payload
            )

        async def write(self, buffer, timeout=None):
            writes.append(buffer)
            if buffer.startswith(b"GET "):
                self.phase = "get"

        async def aclose(self):
            pass

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            import ssl

            assert ssl_context.check_hostname and ssl_context.verify_mode == ssl.CERT_REQUIRED
            tls_names.append(server_hostname)
            return self

        def get_extra_info(self, info):
            return None

    class Backend(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, **kwargs):
            assert host == "proxy.example" and port == 8080
            return Socket()

    async def execute():
        transport = httpx.AsyncHTTPTransport(proxy=proxy)
        # A fake network backend exercises actual CONNECT/TLS orchestration,
        # unlike MockTransport which never runs the HTTPcore TLS path.
        transport._pool._network_backend = Backend()
        async with httpx.AsyncClient(transport=transport) as client:
            return await Downloader(
                settings=configuration(tmp_path, http_proxy=proxy),
                client=client, resolver=public_dns,
            ).download(paper(pdf_url="https://papers.example/a.pdf"))

    result = asyncio.run(execute())
    assert result.status == "downloaded", result.error
    assert tls_names == (["proxy.example", "papers.example"] if proxy.startswith("https:")
                         else ["papers.example"])
    assert any(b"CONNECT 93.184.216.34:443" in value for value in writes)
    assert any(b"Host: papers.example" in value for value in writes)
