"""Unit tests for ``app/services/citation_verification.py`` (Session 10).

Pure function, no I/O, no mocks — construct ``EstimationResult``/``Phase``/
``SourceReference`` instances directly and assert on the returned
``CitationReport``.
"""

from __future__ import annotations

from app.schemas.estimation import EstimationResult, Phase, SourceReference
from app.services.citation_verification import (
    CitationReport,
    CitationStatus,
    verify_citations,
)


def _phase(name: str, *, grounded: bool, sources: list[SourceReference] | None = None) -> Phase:
    return Phase(
        name=name,
        duration_weeks=2,
        cost_eur=5_000,
        summary="Enough summary text to satisfy the min_length constraint.",
        grounded=grounded,
        sources=sources or [],
    )


def _source(chunk_id: str) -> SourceReference:
    return SourceReference(chunk_id=chunk_id, document_id="doc-1", evidence="80h quoted.")


def _result(phases: list[Phase]) -> EstimationResult:
    return EstimationResult(
        summary="A CRM MVP with auth, contacts and a Kanban board for a 5-person team.",
        confidence_pct=80,
        phases=phases,
        total_duration_weeks=1,
        total_cost_eur=1,
    )


def test_all_phases_grounded_with_valid_chunk_ids_are_all_grounded() -> None:
    result = _result(
        [
            _phase("Setup", grounded=True, sources=[_source("101")]),
            _phase("Build", grounded=True, sources=[_source("102"), _source("103")]),
        ]
    )
    report = verify_citations(result, retrieved_chunk_ids={"101", "102", "103"})

    assert len(report.grounded) == 2
    assert len(report.dangling) == 0
    assert len(report.unsupported) == 0
    assert {p.phase_name for p in report.grounded} == {"Setup", "Build"}
    assert all(p.status == CitationStatus.GROUNDED for p in report.grounded)
    assert report.has_dangling_citations is False


def test_phase_citing_unknown_chunk_id_is_dangling_with_offending_id_listed() -> None:
    result = _result(
        [
            _phase("Setup", grounded=True, sources=[_source("101"), _source("does-not-exist")]),
        ]
    )
    report = verify_citations(result, retrieved_chunk_ids={"101"})

    assert len(report.dangling) == 1
    dangling = report.dangling[0]
    assert dangling.phase_name == "Setup"
    assert dangling.status == CitationStatus.DANGLING
    # Only the offending id is listed, not the valid one too.
    assert dangling.dangling_chunk_ids == ["does-not-exist"]
    assert report.has_dangling_citations is True


def test_ungrounded_phase_is_unsupported() -> None:
    result = _result([_phase("Discovery", grounded=False, sources=[])])
    report = verify_citations(result, retrieved_chunk_ids={"101"})

    assert len(report.unsupported) == 1
    assert report.unsupported[0].status == CitationStatus.UNSUPPORTED
    assert report.unsupported[0].dangling_chunk_ids == []
    assert report.has_dangling_citations is False


def test_mixed_report_buckets_all_three_statuses_correctly() -> None:
    result = _result(
        [
            _phase("Setup", grounded=True, sources=[_source("101")]),
            _phase("Build", grounded=True, sources=[_source("does-not-exist")]),
            _phase("Discovery", grounded=False, sources=[]),
        ]
    )
    report = verify_citations(result, retrieved_chunk_ids={"101"})

    assert [p.phase_name for p in report.grounded] == ["Setup"]
    assert [p.phase_name for p in report.dangling] == ["Build"]
    assert [p.phase_name for p in report.unsupported] == ["Discovery"]
    assert report.has_dangling_citations is True
    assert isinstance(report, CitationReport)
