from __future__ import annotations

from quant_agent.tools import paper


def test_no_arxiv_id_is_graceful():
    out = paper.read_paper_text(None)
    assert "No paper on file" in out


def test_fetch_failure_is_graceful(monkeypatch):
    monkeypatch.setattr(paper, "fetch_paper_text", lambda aid, **kw: None)
    out = paper.read_paper_text("2306.00978")
    assert "Could not fetch" in out


def test_full_paper_returned(monkeypatch):
    monkeypatch.setattr(paper, "fetch_paper_text", lambda aid, **kw: "ABSTRACT\n\nfull body text here")
    out = paper.read_paper_text("2306.00978")
    assert "full body text here" in out


def test_section_slice(monkeypatch):
    text = "Intro line\n\nMethod\nwe quantize the weights\n\nExperiments\nresults follow"
    monkeypatch.setattr(paper, "fetch_paper_text", lambda aid, **kw: text)
    out = paper.read_paper_text("x", section="Method")
    assert out.startswith("Method")
    assert "we quantize the weights" in out
    assert "Intro line" not in out


def test_missing_section_falls_back_to_start(monkeypatch):
    monkeypatch.setattr(paper, "fetch_paper_text", lambda aid, **kw: "some body")
    out = paper.read_paper_text("x", section="Nonexistent")
    assert "not found" in out
    assert "some body" in out


def test_truncation(monkeypatch):
    monkeypatch.setattr(paper, "fetch_paper_text", lambda aid, **kw: "x" * 50_000)
    out = paper.read_paper_text("x", max_chars=100)
    assert "truncated" in out
    assert len(out) < 500
