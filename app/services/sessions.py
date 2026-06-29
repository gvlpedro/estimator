"""In-memory session state for multi-turn estimation conversations.

Holds, per ``session_id``:

* a sliding-window ``ConversationHistory`` of messages exchanged with the LLM;
* a structured ``ProjectMetadata`` blob with the facts the assistant has
  gathered (project name, team size, technologies, agreed scope).

Why purely in-process state in this phase
-----------------------------------------
We are still validating the conversational UX and the prompt shape, not running
production traffic. A volatile ``dict`` keyed by ``session_id`` is enough:

* zero infrastructure — no Redis schema, no migrations, no extra container;
* easy to inspect from a REPL and reset between iterations;
* decouples the data model from any serialisation format while it churns.

The tradeoff is explicit and accepted: sessions are lost on process restart,
are not shared across workers, and grow unbounded until the process recycles.
Once the model and the UX stabilise we will move this behind the same
``EstimationCache`` abstraction (Redis-backed) with TTL and eviction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]

DEFAULT_MAX_TURNS = 6
"""Default number of user+assistant PAIRS retained in the sliding window.

A "turn" here is one round-trip with the model: one user message followed by
one assistant message. With ``DEFAULT_MAX_TURNS = 6`` the history holds at
most 12 non-system messages plus the pinned system prompt. Override per call
or globally via the ``MAX_TURNS`` setting (``app/config.py``).
"""


class Message(BaseModel):
    """One message in the conversation, in OpenAI/Anthropic chat format."""

    role: Role
    content: str


class ProjectMetadata(BaseModel):
    """Structured facts the assistant accumulates about the project.

    All fields are optional because metadata grows incrementally turn by turn:
    the first message may only reveal a name, later turns add technologies and
    pin down team size. ``mentioned_technologies`` is a flat list; callers are
    responsible for deduplication if they care about it.
    """

    project_name: str | None = Field(default=None, max_length=120)
    assumed_team_size: int | None = Field(default=None, ge=1, le=100)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = Field(default=None, max_length=4_000)


@dataclass
class ConversationHistory:
    """Sliding-window message log that always preserves the system prompt.

    A "turn" is one user+assistant pair. ``max_turns`` caps how many such
    pairs are kept; once exceeded, the oldest pairs are dropped from the
    front, two messages at a time. The system prompt — when present as the
    first message — is pinned and never evicted, because it carries the
    persona and the static context the rest of the dialogue depends on.
    """

    max_turns: int = DEFAULT_MAX_TURNS
    messages: list[Message] = field(default_factory=list)

    def set_system(self, content: str) -> None:
        """Set or replace the pinned system prompt at position 0."""
        system = Message(role="system", content=content)
        if self.messages and self.messages[0].role == "system":
            self.messages[0] = system
        else:
            self.messages.insert(0, system)

    def add(self, role: Role, content: str) -> None:
        """Append a message and evict the oldest pair(s) if over ``max_turns``."""
        if role == "system":
            self.set_system(content)
            return
        self.messages.append(Message(role=role, content=content))
        self._evict()

    def _evict(self) -> None:
        """Drop oldest user+assistant pairs until under ``max_turns``.

        The system prompt at index 0 is always preserved. We drop in chunks
        of two messages (one full turn) to keep the dialogue well-formed for
        the LLM (alternating user/assistant after the system).

        Example with ``max_turns=2`` (capacity = 4 non-system messages):

            before:  [system, u0, a0, u1, a1, u2, a2]   # 6 non-system, 1 pair over
            after:   [system,         u1, a1, u2, a2]   # oldest pair dropped
        """
        head = 1 if self.messages and self.messages[0].role == "system" else 0
        capacity = self.max_turns * 2
        while len(self.messages) - head > capacity:
            del self.messages[head : head + 2]

    def to_messages_list(self, system_prompt: str | None = None) -> list[dict[str, str]]:
        """Return the chat-completion messages array ready for the LLM API.

        ``system_prompt`` is the freshly-rendered system message. Callers must
        regenerate it from the **current** ``ProjectMetadata`` (and any other
        per-turn context) before invoking this method, so the LLM always sees
        the up-to-date facts on every call. When ``system_prompt`` is
        ``None``, the pinned system message — if any — is reused; pass an
        empty string to drop the system role entirely.

        The non-system messages already stored in ``self.messages`` are
        appended in order, after the sliding-window trim. The pinned system
        message inside ``self.messages`` is never duplicated.
        """
        out: list[dict[str, str]] = []
        if system_prompt is None:
            if self.messages and self.messages[0].role == "system":
                out.append({"role": "system", "content": self.messages[0].content})
        elif system_prompt != "":
            out.append({"role": "system", "content": system_prompt})

        for msg in self.messages:
            if msg.role == "system":
                continue
            out.append({"role": msg.role, "content": msg.content})
        return out


@dataclass
class Session:
    """Per-conversation state: id, history, metadata and last-seen timestamp.

    ``last_resolved_tier`` / ``last_tier_rule`` cache the most recent tier
    resolution so the debug endpoint can surface what the resolver did without
    re-running it. They stay ``None`` until the tier resolver writes to them
    (not wired in this iteration; the field is here so the schema is stable).
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    history: ConversationHistory = field(default_factory=ConversationHistory)
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


class SessionStore:
    """Process-local registry of ``Session`` objects keyed by ``session_id``.

    Volatility is intentional — see the module docstring. Not safe across
    workers; fine for a single-process dev/staging deployment.
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self._sessions: dict[str, Session] = {}
        self.max_turns = max_turns

    def get_or_create(self, session_id: str | None = None) -> Session:
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.touch()
            return session
        session = Session(
            session_id=session_id or str(uuid4()),
            history=ConversationHistory(max_turns=self.max_turns),
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)
