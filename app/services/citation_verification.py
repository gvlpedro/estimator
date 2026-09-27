"""Post-generation citation verification (Session 10).

The ``Phase.grounded`` / ``Phase.sources`` model_validator (in
``app/schemas/estimation.py``) already enforces internal self-consistency:
``grounded=True`` requires at least one ``SourceReference``, and
``grounded=False`` requires none. That validator cannot, however, know
whether a cited ``chunk_id`` actually existed in the retrieved context for
*this* call — Instructor validates the payload in isolation, without access
to the retrieval results. This module closes that gap: it cross-checks every
cited ``chunk_id`` against the set of chunk ids that were genuinely
retrieved, catching a model that fabricates a plausible-looking citation.

Pure function, no I/O, no logging — the caller (``EstimationService``)
decides how to log/report the result. This keeps ``verify_citations``
trivially unit-testable and reusable outside the pipeline (e.g. offline
batch audits over stored results).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.estimation import EstimationResult


class CitationStatus(str, Enum):
    """Outcome of verifying one phase's citations against what was retrieved.

    Matches this codebase's existing convention of ``str, Enum`` subclasses
    for closed sets of values (see ``ProjectType``, ``DetailLevel``,
    ``OutputFormat`` in ``app/schemas/estimation.py``), rather than a bare
    ``Literal`` — these values are also useful as members with ``.value``
    when building dashboards/metrics off logged reports.
    """

    GROUNDED = "grounded"
    DANGLING = "dangling"
    UNSUPPORTED = "unsupported"


class PhaseCitationResult(BaseModel):
    """Verification outcome for a single phase."""

    phase_name: str
    status: CitationStatus
    dangling_chunk_ids: list[str] = Field(default_factory=list)


class CitationReport(BaseModel):
    """Full verification report for one ``EstimationResult``."""

    grounded: list[PhaseCitationResult] = Field(default_factory=list)
    dangling: list[PhaseCitationResult] = Field(default_factory=list)
    unsupported: list[PhaseCitationResult] = Field(default_factory=list)

    @property
    def has_dangling_citations(self) -> bool:
        return len(self.dangling) > 0


def verify_citations(
    estimate: EstimationResult,
    retrieved_chunk_ids: set[str],
) -> CitationReport:
    """Flag any phase whose cited chunk_id was never in the retrieved context.

    Per-phase logic:

    - ``grounded=False`` -> ``UNSUPPORTED``. This is the honest, correct
      case: a phase can legitimately have no supporting evidence. The
      schema-level validator already guarantees ``sources`` is empty here,
      so there is nothing left to check against ``retrieved_chunk_ids``.
    - ``grounded=True`` -> check every ``source.chunk_id`` against
      ``retrieved_chunk_ids``. All present -> ``GROUNDED``. Any missing ->
      ``DANGLING``, with ``dangling_chunk_ids`` listing exactly the
      offending ids (not the valid ones too).
    """
    report = CitationReport()

    for phase in estimate.phases:
        if not phase.grounded:
            report.unsupported.append(
                PhaseCitationResult(phase_name=phase.name, status=CitationStatus.UNSUPPORTED)
            )
            continue

        offending = [s.chunk_id for s in phase.sources if s.chunk_id not in retrieved_chunk_ids]
        if offending:
            report.dangling.append(
                PhaseCitationResult(
                    phase_name=phase.name,
                    status=CitationStatus.DANGLING,
                    dangling_chunk_ids=offending,
                )
            )
        else:
            report.grounded.append(
                PhaseCitationResult(phase_name=phase.name, status=CitationStatus.GROUNDED)
            )

    return report
