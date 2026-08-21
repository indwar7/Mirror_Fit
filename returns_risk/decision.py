"""The decision contract.

A flag is a request for human attention, never a verdict. The system can
send a return to a reviewer; it cannot deny a refund. That is enforced
here structurally rather than by convention:

  * `Action` is a closed enumeration with exactly two members, so there is
    no denial value available to return even by mistake.
  * `_assert_no_denial_action()` runs at import and fails loudly if anyone
    later adds one.
  * `tests.py` asserts that no input — including adversarial confidences,
    NaN, and out-of-range scores — can produce anything else.

The practical consequence: a false positive costs a genuine customer a
review step and some delay. It never costs them their refund. That is what
keeps the false-positive cost bounded to friction, and it is the reason the
5:1 weighting is a defensible stance rather than a way to hide a harsh
policy behind a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class Action(StrEnum):
    """The only two outcomes this system can produce."""

    APPROVE_REFUND = "approve_refund"
    ROUTE_TO_REVIEW = "route_to_review"


#: Substrings that must never appear in an action value. An automated
#: denial is out of scope for this component by design.
FORBIDDEN_ACTION_TOKENS = ("deny", "reject", "block", "refuse", "decline")


def _assert_no_denial_action() -> None:
    for member in Action:
        value = member.value.lower()
        for token in FORBIDDEN_ACTION_TOKENS:
            if token in value:
                raise AssertionError(
                    f"Action.{member.name} = {member.value!r} looks like an automated "
                    "denial. This component routes to human review and must not deny "
                    "refunds. See returns_risk/decision.py."
                )


_assert_no_denial_action()


@dataclass(frozen=True)
class Decision:
    """Result of scoring a single return.

    `confidence` is the model's P(fraud) for this return, in [0, 1]. It is
    reported whether or not the return was flagged, so a reviewer can tell
    a borderline case from an emphatic one. It is deliberately *not*
    "confidence that the decision is correct" — that would be a different
    and much stronger claim than this model can support.
    """

    flagged: bool
    confidence: float
    action: Action
    threshold: float = 0.5
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        """The wire format specified by the brief."""
        return {
            "flagged": self.flagged,
            "confidence": round(self.confidence, 4),
            "action": self.action.value,
        }

    def to_review_payload(self) -> dict[str, Any]:
        """Richer form for the review queue. A reviewer who cannot see why a
        case was flagged cannot overturn it competently, so the reason codes
        travel with it."""
        return {
            **self.to_payload(),
            "threshold": round(self.threshold, 4),
            "reason_codes": list(self.reason_codes),
        }


def _clamp_probability(value: float) -> float:
    """Coerce anything unusable into a safe score.

    A NaN or out-of-range score must not crash the returns flow, and must
    not silently become a high-risk score either. Unusable input resolves
    to 0.0, which routes the customer down the normal approval path — the
    failure mode that does not punish them.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return min(1.0, max(0.0, number))


def explain(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Human-readable reason codes for the review queue.

    These describe what is unusual about the case. They are not the model's
    internal coefficients and must not be presented to a reviewer as proof.
    """
    codes: list[str] = []

    reason = row.get("return_reason")
    used_tryon = bool(row.get("used_tryon", False))
    mismatch = row.get("fit_mismatch_score")
    has_mismatch = mismatch is not None and not (
        isinstance(mismatch, float) and math.isnan(mismatch)
    )

    if reason == "doesnt_fit" and used_tryon and has_mismatch and mismatch < 0.25:
        codes.append("FIT_CLAIM_CONTRADICTS_TRYON")
    if reason == "doesnt_fit" and not used_tryon:
        codes.append("FIT_CLAIM_WITHOUT_TRYON")
    if float(row.get("prior_return_rate", 0.0) or 0.0) >= 0.5:
        codes.append("HIGH_PRIOR_RETURN_RATE")
    if float(row.get("days_since_purchase", 0) or 0) > 28:
        codes.append("LATE_IN_RETURN_WINDOW")
    if float(row.get("order_value_inr", 0) or 0) >= 6000:
        codes.append("HIGH_VALUE_ORDER")
    if float(row.get("account_age_days", 9999) or 9999) <= 60:
        codes.append("NEW_ACCOUNT")
    if reason in ("not_as_described", "damaged"):
        codes.append(f"REASON_{str(reason).upper()}")

    # An exculpatory code matters as much as an incriminating one: it tells
    # the reviewer the try-on actually predicted a poor fit.
    if reason == "doesnt_fit" and has_mismatch and mismatch >= 0.55:
        codes.append("FIT_CLAIM_SUPPORTED_BY_TRYON")

    return tuple(codes)


def decide(
    probability: float,
    threshold: float,
    row: Mapping[str, Any] | None = None,
) -> Decision:
    """Turn a fraud probability into an action.

    Only ever `approve_refund` or `route_to_review`.
    """
    confidence = _clamp_probability(probability)
    cut = _clamp_probability(threshold)

    flagged = confidence >= cut
    action = Action.ROUTE_TO_REVIEW if flagged else Action.APPROVE_REFUND

    return Decision(
        flagged=flagged,
        confidence=confidence,
        action=action,
        threshold=cut,
        reason_codes=explain(row) if (flagged and row is not None) else tuple(),
    )
