from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.acb import BossTrace

PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]
ExampleFormat = Literal["markdown", "json", "narrative"]


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class ReferenceProject(BaseModel):
    """A past project the caller wants the model to use as a similarity anchor."""

    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=10, max_length=600)
    total_hours: int = Field(ge=1)
    total_cost_eur: int = Field(ge=0)
    notes: str | None = Field(default=None, max_length=400)


class EstimationRequest(BaseModel):
    """Typed payload sent by the business backend or the Streamlit form.

    ``description`` accepts up to 80 000 chars so the same field can carry a
    raw meeting transcription. The estimator never separates "description" and
    "transcription" — both feed the prompt verbatim.
    """

    description: str = Field(
        min_length=20,
        max_length=80_000,
        description="Free-text description or transcription of the project to estimate.",
    )
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    reference_projects: list[ReferenceProject] | None = None


# ---------------------------------------------------------------------------
# Structured estimation result (Session 4)
# ---------------------------------------------------------------------------


OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30


class Phase(BaseModel):
    """One phase in the breakdown of an estimation."""

    name: str = Field(min_length=1, max_length=64)
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0, le=1_000_000)
    summary: str = Field(min_length=10, max_length=600)


class EstimationResult(BaseModel):
    """Validated structured estimation returned by the LLM via Instructor.

    Field order is deliberate: ``phases`` comes BEFORE the totals so the LLM
    commits to per-phase numbers first (autoregressive generation) and only
    needs to sum them when filling the totals. With totals first, small models
    (e.g. ``gpt-4o-mini``) pick a round number and back-fit phases badly.
    """

    summary: str = Field(min_length=10, max_length=1200)
    # Nivel de certeza del modelo sobre la estimación, de 0 a 100.
    # Se calcula a partir de la claridad del scope, integraciones definidas y restricciones explícitas; <30 obliga a prefijar el summary con "Out of scope:".
    confidence_pct: int = Field(ge=0, le=100)
    # Desglose del proyecto en fases (1 a 8), cada una con nombre, duración, coste y resumen.
    # Se construye dividiendo el alcance en bloques de trabajo coherentes antes de calcular los totales.
    phases: list[Phase] = Field(min_length=1, max_length=8)
    # Duración total del proyecto en semanas.
    # Se calcula como la suma de duration_weeks de todas las fases (asumiendo ejecución secuencial).
    total_duration_weeks: int = Field(ge=1, le=104)
    total_cost_eur: int = Field(ge=0, le=2_000_000)

    @model_validator(mode="before")
    @classmethod
    def reconcile_totals_from_phases(cls, data: Any) -> Any:
        # Las fases son la fuente de verdad: los totales se derivan de su suma
        # para evitar que un error de aritmética del LLM (p.ej. fases = 12_000
        # EUR pero total = 13_000) gaste reintentos de Instructor y termine en
        # 502. El Critic sigue detectando incoherencias semánticas de mayor
        # nivel via `math_error`.
        if not isinstance(data, dict):
            return data
        phases = data.get("phases")
        if not isinstance(phases, list) or not phases:
            return data
        cost_sum = 0
        weeks_sum = 0
        for phase in phases:
            if not isinstance(phase, dict):
                return data
            cost_sum += int(phase.get("cost_eur", 0) or 0)
            weeks_sum += int(phase.get("duration_weeks", 0) or 0)
        data["total_cost_eur"] = cost_sum
        data["total_duration_weeks"] = weeks_sum
        return data

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to "
                f"start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the "
                f"description is too vague to size"
            )
        return self


class EstimationResponse(BaseModel):
    """Structured response: validated result + prompt version + cache flag."""

    result: EstimationResult
    prompt_version: str
    cached: bool = False


class ACBResponse(BaseModel):
    """Actor-Critic-Boss response: same result envelope as ``EstimationResponse``
    plus the orchestration trace so the caller can render the audit panel.
    """

    result: EstimationResult
    prompt_version: str
    cached: bool = False
    acb: BossTrace


# ---------------------------------------------------------------------------
# Streaming endpoint + evaluation helpers (kept from earlier sessions)
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token consumption details from the LLM call(s)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    preprocessing_input_tokens: int = 0
    preprocessing_output_tokens: int = 0


class StructureCheck(BaseModel):
    """Level-1 structural evaluation of a generated estimation."""

    has_title: bool
    has_breakdown_table: bool
    has_totals_section: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: int | None
    sum_row_hours: int | None
    hours_match: bool | None
    declared_total_cost: float | None
    sum_row_cost: float | None
    cost_match: bool | None
    finish_reason_ok: bool
    score: float
    issues: list[str]


class StreamEstimationRequest(BaseModel):
    """Streaming endpoint request — only the transcription, knobs are not exposed."""

    transcription: str = Field(..., min_length=50, description="Meeting transcription text")
    model: str | None = Field(default=None, description="Override the default model")
    max_tokens: int = Field(default=4000, gt=0, le=16000)
