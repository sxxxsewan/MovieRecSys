"""Stable user assignment for offline experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable


def assign_group(
    user_id: Hashable,
    experiment_name: str,
    control: str = "control",
    treatment: str = "treatment",
    treatment_fraction: float = 0.5,
) -> str:
    """Assign a user deterministically using a stable hash of user and experiment."""
    if not experiment_name:
        raise ValueError("experiment_name must not be empty")
    if not control or not treatment or control == treatment:
        raise ValueError("control and treatment must be distinct nonempty names")
    if not 0.0 < treatment_fraction < 1.0:
        raise ValueError("treatment_fraction must be strictly between 0 and 1")
    identity = f"{experiment_name}\0{type(user_id).__qualname__}\0{user_id!s}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") / 2**64
    return treatment if bucket < treatment_fraction else control
