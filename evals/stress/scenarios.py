"""Three conversational stress profiles + the fact tracker that feeds the
``MemoryDriftMetric``.

Each profile defines 20 turns over the same project. The fact tracker pins,
per turn, the strings that the system MUST still know about on a later
turn — the metric reads the snapshot at turn N and checks every fact
introduced at turn k ≤ N.

The data is intentionally pure-Python: no IO, no network. The runner in
``run.py`` slices by ``N ∈ {1, 3, 6, 10, 20}``, executes each turn against the
sessions endpoint, and pairs the resulting snapshot with the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal


AssertionKind = Literal["contains", "forbidden"]


SUPPORTED_TURN_LENGTHS: tuple[int, ...] = (1, 3, 6, 10, 20)


@dataclass(frozen=True)
class FactAssertion:
    """Pin a fact to a turn and to where it should be visible.

    ``introduced_at_turn`` is the 1-based index of the turn that first
    introduced the fact. ``kind`` is either ``"contains"`` (the value MUST be
    found in at least one of the ``where`` fields) or ``"forbidden"`` (the
    value MUST NOT appear; used to assert that pivots cleanly drop the old
    stack). ``superseded_at_turn`` is set on facts that get replaced by a
    later turn — the runner uses it to relax the assertion past that point.
    """

    fact: str
    kind: AssertionKind
    introduced_at_turn: int
    where: tuple[str, ...] = ("summary", "anchors", "metadata")
    label: str = ""
    superseded_at_turn: int | None = None


@dataclass(frozen=True)
class StressTurn:
    """One conversational turn: the user message + its own assertions."""

    index: int  # 1-based
    transcript: str
    assertions: tuple[FactAssertion, ...] = ()


@dataclass(frozen=True)
class StressScenario:
    """A profile sliced to a particular length.

    ``profile`` and ``project_type`` are forwarded to the sessions endpoint.
    ``turns`` and ``assertions`` are the slice; the latter is the union of
    every per-turn assertion that should still hold at the end of the slice.
    """

    profile: str
    project_type: str
    turns: tuple[StressTurn, ...]
    assertions: tuple[FactAssertion, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Profile 1 — Growing project (Nimbus CRM, B2B SaaS)
# ---------------------------------------------------------------------------
#
# Turn 1 sets the project_name. Subsequent turns add coherent requirements:
# authentication, multi-tenant, audit log, CSV export, SSO, billing, etc.
# After turn 20 we still expect "Nimbus" to be remembered AND the latest
# additions to appear in metadata.mentioned_technologies / agreed_scope.


def growing_project() -> list[StressTurn]:
    """20-turn build-up around the same project."""

    def turn(idx: int, transcript: str, *, asserts: tuple[FactAssertion, ...]) -> StressTurn:
        return StressTurn(index=idx, transcript=transcript, assertions=asserts)

    fa = FactAssertion

    return [
        turn(
            1,
            "We're scoping a new B2B SaaS called Nimbus — a customer-success CRM. "
            "Initial scope: contacts, organisations, and a deal pipeline. "
            "Stack will be React on the front and Postgres on the back.",
            asserts=(
                fa("Nimbus", "contains", 1, label="project_name"),
                fa("React", "contains", 1, label="frontend_tech"),
                fa("Postgres", "contains", 1, label="db_tech"),
            ),
        ),
        turn(
            2,
            "Add authentication: email + password to start, with passwordless magic "
            "links on the roadmap. Sessions stored in Postgres for now.",
            asserts=(fa("authentication", "contains", 2, label="auth_feature"),),
        ),
        turn(
            3,
            "It must be multi-tenant — each customer organisation lives in its own "
            "tenant. Tenant id propagated on every row.",
            asserts=(fa("multi-tenant", "contains", 3, label="tenancy"),),
        ),
        turn(
            4,
            "Compliance now wants an audit log of every mutation: who, what, when, "
            "old value, new value. Append-only.",
            asserts=(fa("audit log", "contains", 4, label="audit"),),
        ),
        turn(
            5,
            "Add CSV export of the deal pipeline. Streamed download, no in-memory "
            "buffering for large tenants.",
            asserts=(fa("CSV export", "contains", 5, label="csv_export"),),
        ),
        turn(
            6,
            "Sales now want SSO via SAML and OIDC. Okta and Google Workspace as the "
            "first two IdPs.",
            asserts=(
                fa("SSO", "contains", 6, label="sso"),
                fa("Okta", "contains", 6, label="sso_okta"),
            ),
        ),
        turn(
            7,
            "Billing: monthly subscription per seat, Stripe Billing. Trials of 14 days "
            "and proration on upgrades.",
            asserts=(fa("Stripe", "contains", 7, label="billing_tech"),),
        ),
        turn(
            8,
            "Notification centre: in-app + email digests. Email transactional via "
            "Postmark, digests scheduled in a worker queue.",
            asserts=(fa("Postmark", "contains", 8, label="email_provider"),),
        ),
        turn(
            9,
            "Reporting dashboard with cohort retention, conversion funnel and revenue "
            "by tenant. Live filters by date and pipeline stage.",
            asserts=(fa("dashboard", "contains", 9, label="reporting"),),
        ),
        turn(
            10,
            "Mobile companion app — read-only for now: pipeline view, contact details, "
            "push notifications on deal updates.",
            asserts=(fa("mobile", "contains", 10, label="mobile_app"),),
        ),
        turn(
            11,
            "Public REST API with API keys per tenant and per-key rate limits. "
            "OpenAPI 3 schema published.",
            asserts=(fa("REST API", "contains", 11, label="public_api"),),
        ),
        turn(
            12,
            "Add webhooks: tenants subscribe to deal_won, contact_created and "
            "audit_event. Signed with HMAC.",
            asserts=(fa("webhooks", "contains", 12, label="webhooks"),),
        ),
        turn(
            13,
            "Search across contacts, organisations and deals — fuzzy, with type-ahead. "
            "Build it on Postgres trigram first; we can move to Elasticsearch later.",
            asserts=(fa("search", "contains", 13, label="search_feature"),),
        ),
        turn(
            14,
            "Bulk import: CSV upload of contacts with column mapping and a dry-run "
            "preview before commit.",
            asserts=(fa("bulk import", "contains", 14, label="import_feature"),),
        ),
        turn(
            15,
            "Custom fields per tenant — text, number, single-select. Stored as JSONB.",
            asserts=(fa("custom fields", "contains", 15, label="custom_fields"),),
        ),
        turn(
            16,
            "Activity timeline on each contact: emails sent, calls logged, deals "
            "moved, audit entries. Reverse-chronological.",
            asserts=(fa("activity timeline", "contains", 16, label="timeline"),),
        ),
        turn(
            17,
            "Role-based permissions: owner, admin, member, viewer. Per-pipeline "
            "overrides.",
            asserts=(fa("permissions", "contains", 17, label="rbac"),),
        ),
        turn(
            18,
            "Data residency: tenants can opt into EU-only storage. AWS Frankfurt as "
            "the EU region.",
            asserts=(fa("data residency", "contains", 18, label="residency"),),
        ),
        turn(
            19,
            "Add an admin console for the support team: impersonation with audit, "
            "tenant suspension, manual refunds.",
            asserts=(fa("admin console", "contains", 19, label="admin_console"),),
        ),
        turn(
            20,
            "Finally — GDPR right-to-be-forgotten: a delete request that purges PII "
            "from active tables and from the audit log within the retention window.",
            asserts=(
                fa("GDPR", "contains", 20, label="gdpr"),
                # The original project name MUST still be present at turn 20.
                fa("Nimbus", "contains", 1, label="project_name_persistence"),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Profile 2 — Pivoting project (React → Flutter at turn 5)
# ---------------------------------------------------------------------------


def pivoting_project() -> list[StressTurn]:
    """20-turn flow where turn 5 swaps the entire client stack.

    From turn 5 onward, ``React`` and ``Next.js`` must NOT appear in metadata
    or in the assistant summary — they have been explicitly dropped. ``Flutter``
    must appear instead.
    """

    def turn(idx: int, transcript: str, *, asserts: tuple[FactAssertion, ...]) -> StressTurn:
        return StressTurn(index=idx, transcript=transcript, assertions=asserts)

    fa = FactAssertion

    return [
        turn(
            1,
            "New product: Atlas — a marketplace for freelance interior designers. "
            "Web first, React + Next.js on the front, Postgres on the back.",
            asserts=(
                fa("Atlas", "contains", 1, label="project_name"),
                fa("React", "contains", 1, label="frontend_initial", superseded_at_turn=5),
                fa("Next.js", "contains", 1, label="ssr_initial", superseded_at_turn=5),
            ),
        ),
        turn(
            2,
            "Designers list their services with photos, availability and prices. "
            "Customers browse, filter by style, and request quotes.",
            asserts=(),
        ),
        turn(
            3,
            "Payments via Stripe Connect for marketplace splits. KYC on designers.",
            asserts=(fa("Stripe", "contains", 3, label="payments_tech"),),
        ),
        turn(
            4,
            "Booking flow: tentative request, designer confirms within 24h, deposit "
            "captured at confirmation, balance at completion.",
            asserts=(fa("booking", "contains", 4, label="booking_flow"),),
        ),
        turn(
            5,
            "Important pivot — we are dropping the web-first stack. Going mobile-only "
            "with Flutter for both iOS and Android. Forget React and Next.js, neither "
            "will ship. The backend stays the same.",
            asserts=(
                fa("Flutter", "contains", 5, label="frontend_pivot"),
                fa("React", "forbidden", 5, label="frontend_pivot_forbidden_react"),
                fa("Next.js", "forbidden", 5, label="frontend_pivot_forbidden_next"),
            ),
        ),
        turn(
            6,
            "Push notifications: Firebase Cloud Messaging — both platforms.",
            asserts=(fa("Firebase", "contains", 6, label="push_tech"),),
        ),
        turn(
            7,
            "Authentication: Firebase Auth too — email + Google + Apple Sign-In.",
            asserts=(fa("Apple Sign-In", "contains", 7, label="auth_apple"),),
        ),
        turn(
            8,
            "Designers can upload portfolios from their phones. Image pipeline must "
            "compress on-device before upload to S3.",
            asserts=(fa("S3", "contains", 8, label="storage_tech"),),
        ),
        turn(
            9,
            "Chat between customer and designer once a quote is accepted. End-to-end "
            "encrypted at rest.",
            asserts=(fa("chat", "contains", 9, label="chat_feature"),),
        ),
        turn(
            10,
            "Reviews and ratings post-booking, 5-star with optional photos.",
            asserts=(fa("reviews", "contains", 10, label="reviews_feature"),),
        ),
        turn(
            11,
            "Admin web console for our ops team — minimal, just dispute handling and "
            "designer onboarding queue.",
            asserts=(fa("admin", "contains", 11, label="admin_feature"),),
        ),
        turn(
            12,
            "Localisation in English, Spanish and French — Flutter intl bundles "
            "shipped with the app, not over-the-air.",
            asserts=(fa("localisation", "contains", 12, label="i18n"),),
        ),
        turn(
            13,
            "Analytics: Mixpanel for product, Sentry for crash and error reporting.",
            asserts=(fa("Mixpanel", "contains", 13, label="analytics_tech"),),
        ),
        turn(
            14,
            "Search with filters: style, budget range, location, availability window. "
            "Backed by a Postgres full-text index.",
            asserts=(fa("search", "contains", 14, label="search_feature"),),
        ),
        turn(
            15,
            "Designers can offer recurring bookings — monthly visits, weekly check-ins.",
            asserts=(fa("recurring", "contains", 15, label="recurring_bookings"),),
        ),
        turn(
            16,
            "Tax handling per region — VAT for EU, sales tax for US states.",
            asserts=(fa("VAT", "contains", 16, label="tax_rules"),),
        ),
        turn(
            17,
            "Push notification preferences: customers and designers each pick which "
            "events ping them.",
            asserts=(fa("preferences", "contains", 17, label="push_prefs"),),
        ),
        turn(
            18,
            "Designer payouts: weekly Stripe transfers, with a hold-back during "
            "dispute windows.",
            asserts=(fa("payouts", "contains", 18, label="payouts"),),
        ),
        turn(
            19,
            "GDPR data export and deletion for customers AND designers. Account "
            "closure should keep historical reviews anonymised.",
            asserts=(fa("GDPR", "contains", 19, label="gdpr"),),
        ),
        turn(
            20,
            "Roadmap close-out: ship the Flutter MVP for both stores by Q4, then "
            "decide on a web companion next year.",
            asserts=(
                fa("Flutter", "contains", 5, label="frontend_pivot_persistence"),
                fa("React", "forbidden", 5, label="forbidden_react_at_t20"),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Profile 3 — Contradicting project (budget 30k€ at turn 3, 80k€ at turn 8)
# ---------------------------------------------------------------------------


def contradicting_project() -> list[StressTurn]:
    """20-turn flow with an explicit budget contradiction.

    Turn 3 states 30k€; turn 8 raises it to 80k€. The metric must check that
    after turn 8 the LATEST value is what survives in the most authoritative
    surface (metadata.agreed_scope or summary). The earlier value is allowed
    to live on as a transcript anchor — only the *active* fact matters.
    """

    def turn(idx: int, transcript: str, *, asserts: tuple[FactAssertion, ...]) -> StressTurn:
        return StressTurn(index=idx, transcript=transcript, assertions=asserts)

    fa = FactAssertion

    return [
        turn(
            1,
            "Onboarding portal for our 200-person engineering org. We want it to "
            "replace the spreadsheet we use today.",
            asserts=(fa("onboarding", "contains", 1, label="project_kind"),),
        ),
        turn(
            2,
            "Single sign-on with Okta, a checklist per role, document upload, signing "
            "via DocuSign.",
            asserts=(
                fa("Okta", "contains", 2, label="sso_tech"),
                fa("DocuSign", "contains", 2, label="signing_tech"),
            ),
        ),
        turn(
            3,
            "Budget context: we have about 30k€ to spend on this in the current "
            "quarter. Keep that as the working number.",
            asserts=(
                fa(
                    "30",
                    "contains",
                    3,
                    label="budget_initial",
                    superseded_at_turn=8,
                ),
            ),
        ),
        turn(
            4,
            "Two developers internal, plus we'll bring a contract designer for 2 weeks.",
            asserts=(fa("two developers", "contains", 4, label="team_size"),),
        ),
        turn(
            5,
            "Audit trail of every signed document, retained for 7 years per HR policy.",
            asserts=(fa("audit trail", "contains", 5, label="audit_trail"),),
        ),
        turn(
            6,
            "Slack notifications on completed checklists, with a digest summary on "
            "Mondays.",
            asserts=(fa("Slack", "contains", 6, label="notifications_tech"),),
        ),
        turn(
            7,
            "We want mobile-friendly screens, not a native app. Web only, responsive.",
            asserts=(fa("responsive", "contains", 7, label="responsive"),),
        ),
        turn(
            8,
            "Update on budget: legal and HR agreed to fund this through their own "
            "ledgers too. New budget is 80k€, not 30k€ — the original number is "
            "stale, treat 80k as the binding figure from now on.",
            asserts=(
                fa("80", "contains", 8, label="budget_current"),
            ),
        ),
        turn(
            9,
            "Add role-aware checklists: engineers, designers, PMs, sales each get a "
            "different default flow.",
            asserts=(fa("role-aware", "contains", 9, label="role_aware"),),
        ),
        turn(
            10,
            "Manager review step: each completed checklist gets a manager sign-off "
            "before it counts as done.",
            asserts=(fa("manager review", "contains", 10, label="manager_review"),),
        ),
        turn(
            11,
            "Templates: HR can author templates without engineering involvement. "
            "Markdown editor with variable substitution.",
            asserts=(fa("templates", "contains", 11, label="templates"),),
        ),
        turn(
            12,
            "Reporting: time-to-onboarding median, drop-off by step, signed-document "
            "counts. Exportable as PDF.",
            asserts=(fa("reporting", "contains", 12, label="reporting"),),
        ),
        turn(
            13,
            "Integration with our HRIS — BambooHR — to pull employee records and "
            "trigger onboarding flows on hire.",
            asserts=(fa("BambooHR", "contains", 13, label="hris_tech"),),
        ),
        turn(
            14,
            "Multilingual: English and Spanish, since our Madrid office is growing.",
            asserts=(fa("Spanish", "contains", 14, label="i18n"),),
        ),
        turn(
            15,
            "Reminder cadence: 24h, 72h, 5 days. Email plus Slack ping.",
            asserts=(fa("reminders", "contains", 15, label="reminders"),),
        ),
        turn(
            16,
            "Document templates need version history — auditors must reconstruct "
            "what a template said on any given date.",
            asserts=(fa("version history", "contains", 16, label="version_history"),),
        ),
        turn(
            17,
            "Self-serve unblocking: a new hire who is stuck can ping their buddy "
            "via a button in the checklist UI.",
            asserts=(fa("buddy", "contains", 17, label="buddy_feature"),),
        ),
        turn(
            18,
            "Accessibility: WCAG AA across all screens.",
            asserts=(fa("WCAG", "contains", 18, label="a11y"),),
        ),
        turn(
            19,
            "Data retention policy: signed PDFs stay 7 years, transient checklists "
            "purged after 18 months.",
            asserts=(fa("retention", "contains", 19, label="retention"),),
        ),
        turn(
            20,
            "Closing scope: ship by Q4 within the 80k€ envelope we agreed earlier. "
            "Internal sponsor signs off this week.",
            asserts=(
                fa("80", "contains", 8, label="budget_current_persistence"),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Registry + slicing
# ---------------------------------------------------------------------------


PROFILES: dict[str, Callable[[], list[StressTurn]]] = {
    "growing": growing_project,
    "pivot": pivoting_project,
    "contradiction": contradicting_project,
}


PROFILE_PROJECT_TYPES: dict[str, str] = {
    "growing": "web_saas",
    "pivot": "mobile_app",      # post-pivot it's a mobile project
    "contradiction": "internal_tool",
}


def scenarios_for_length(profile: str, n: int) -> StressScenario:
    """Return the first ``n`` turns of ``profile`` plus the union of every
    assertion that should be evaluable at the END of the slice.

    An assertion is included if its ``introduced_at_turn <= n``. The runner
    chooses how to handle ``superseded_at_turn`` — typically the metric
    treats a superseded fact as "best-effort" past its supersession turn."""
    if profile not in PROFILES:
        raise ValueError(
            f"unknown profile {profile!r}; choose one of {sorted(PROFILES)}"
        )
    if n not in SUPPORTED_TURN_LENGTHS:
        raise ValueError(
            f"unsupported turn length {n}; choose one of {SUPPORTED_TURN_LENGTHS}"
        )
    all_turns = PROFILES[profile]()
    sliced = tuple(all_turns[:n])
    assertions: list[FactAssertion] = []
    for t in sliced:
        for a in t.assertions:
            if a.introduced_at_turn <= n:
                assertions.append(a)
    return StressScenario(
        profile=profile,
        project_type=PROFILE_PROJECT_TYPES[profile],
        turns=sliced,
        assertions=tuple(assertions),
    )
