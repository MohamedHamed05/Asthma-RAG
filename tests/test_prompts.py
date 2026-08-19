"""Lock the clinical system prompt to prevent accidental drift."""

import hashlib

from asthma_rag.prompts import SYSTEM_PROMPT

# SHA-256 of the verbatim clinical system prompt from hackathon (2).py lines 254-643.
# Any change to SYSTEM_PROMPT must be an explicit, deliberate action.
EXPECTED_SHA256 = "182a1054909d4924cece001449c1f825fc2699b87c57b63eab23e97f9578f702"


def test_system_prompt_exported() -> None:
    """SYSTEM_PROMPT is exported as a non-empty string."""
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0


def test_system_prompt_sha256() -> None:
    """SYSTEM_PROMPT byte content matches the recorded SHA-256 hash.

    This is the clinical-safety lock: any edit to the prompt text will
    flip this hash and cause a deliberate, reviewable failure.
    """
    actual = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert actual == EXPECTED_SHA256, (
        f"SYSTEM_PROMPT hash mismatch — prompt text was modified.\n"
        f"  Expected: {EXPECTED_SHA256}\n"
        f"  Actual:   {actual}\n"
        f"If this change is intentional, update EXPECTED_SHA256 in this file."
    )
