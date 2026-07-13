"""Strict codec for the isolated backend worker JSON-lines protocol.

Protocol 2.0.0 adds a mandatory worker-first identity handshake. The codec rejects protocol drift,
wrong-direction messages, unknown fields, and a message type whose body does not validate against the
canonical contract map. It is intentionally dependency-light: stdlib + pydantic contracts, no torch.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import ValidationError

from corpus_studio.platform.common import ContractModel, JsonObject
from corpus_studio.platform.contracts import (
    WORKER_BODY_BY_TYPE,
    WORKER_PROTOCOL_VERSION,
    WorkerBody,
    WorkerMessage,
    WorkerMessageType,
)

PROTOCOL_VERSION = WORKER_PROTOCOL_VERSION


class WorkerProtocolError(ValueError):
    """The peer violated the worker wire contract or negotiated the wrong protocol version."""


def build_worker_message(
    message_type: WorkerMessageType,
    body: WorkerBody | JsonObject,
    *,
    message_id: str,
    direction: Literal["core_to_worker", "worker_to_core"],
    correlation_id: str | None = None,
    sent_at: str | None = None,
) -> WorkerMessage:
    """Build and fully validate one protocol-v2 envelope."""

    payload = body.model_dump(mode="json") if isinstance(body, ContractModel) else body
    return WorkerMessage.model_validate(
        {
            "protocol_version": PROTOCOL_VERSION,
            "message_id": message_id,
            "correlation_id": correlation_id,
            "direction": direction,
            "sent_at": sent_at,
            "type": message_type,
            "body": payload,
        }
    )


def encode_worker_message(message: WorkerMessage) -> str:
    """Encode one validated envelope as a compact single JSON line (without the newline)."""

    return message.model_dump_json(exclude_none=True)


def decode_worker_message(
    line: str,
    *,
    expected_direction: Literal["core_to_worker", "worker_to_core"] | None = None,
) -> WorkerMessage:
    """Decode one line and enforce protocol version, direction, envelope, and typed body."""

    if not line.strip():
        raise WorkerProtocolError("blank protocol line")
    message_type: object = "unknown"
    try:
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise TypeError("protocol envelope must be a JSON object")
        message_type = raw.get("type", "unknown")
        raw_version = raw.get("protocol_version")
        if raw_version != PROTOCOL_VERSION:
            raise WorkerProtocolError(
                f"protocol version {raw_version!r} != required {PROTOCOL_VERSION!r}"
            )
        message = WorkerMessage.model_validate(raw)
    except WorkerProtocolError:
        raise
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
        raise WorkerProtocolError(
            f"invalid WorkerMessage type {message_type!r}: {exc}"
        ) from exc
    if expected_direction is not None and message.direction != expected_direction:
        raise WorkerProtocolError(
            f"message direction {message.direction!r} != expected {expected_direction!r}"
        )
    return message


def parse_worker_body(message: WorkerMessage) -> ContractModel:
    """Return the canonical typed body selected by ``message.type``."""

    try:
        expected = WORKER_BODY_BY_TYPE[message.type]
        if isinstance(message.body, expected):
            return message.body
        return expected.model_validate(message.body.model_dump(mode="json"))
    except (KeyError, ValidationError, ValueError, TypeError) as exc:
        raise WorkerProtocolError(
            f"invalid {message.type!r} message body: {exc}"
        ) from exc
