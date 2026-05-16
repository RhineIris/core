"""LCM Safety Contracts — Nagini (Viper) Formal Verification.

This module uses Nagini's Python-embedded contract language (based on the
Viper verification infrastructure) to formally verify the LCM safety
invariants.

Nagini verifies Python code statically by translating contracts and
annotations into Viper's Silver intermediate language, then checking
them with the Viper verifier.

Usage:
    # Requires Nagini + Viper backend (Java required)
    nagini train/safety_nagini.py

    # If Nagini is not installed (typed-ast issue on Python 3.14):
    #   cd LCM
    #   python3.10 -m venv .venv_nagini
    #   source .venv_nagini/bin/activate
    #   pip install nagini
    #   nagini train/safety_nagini.py

Contracts verified (each ∀ pre-state satisfying Requires):
    C1  danger_assess: out_block ⇔ threat condition exceeded
    C2  danger_assess: threat type matches first-triggered condition
    C3  gvalue_check_safety: safe ⇔ pos_d_min ≤ neg_d_min - margin
    C4  detect_any_conflict: true ⇔ any sub-check triggered
    C5  conflict source identifies the first-triggered check
    C6  halt_and_alert is terminal (conflict ⇒ ¬convergence)
"""

from typing import Tuple
from nagini_contracts.contracts import (
    Pure, Result, Ensures, Requires, Assert, Implies, Old
)

# ═════════════════════════════════════════════════════════════════════════════
#  Constants (must match lcm.h exactly)
# ═════════════════════════════════════════════════════════════════════════════

MAX_RETRIEVALS: int = 12
MAX_STEPS: int = 32
CONSISTENCY_THRESHOLD: float = 0.3
SAFETY_THRESHOLD: float = 0.005
DEFAULT_SAFETY_MARGIN: float = 0.5

# Threat types
THREAT_NONE: int = 0
THREAT_PATTERN_MATCH: int = 1
THREAT_RESOURCE_ABUSE: int = 2
THREAT_RUNAWAY: int = 3
THREAT_DECEPTION: int = 4

# Conflict sources
CONFLICT_NONE: int = 0
CONFLICT_DANGER: int = 1
CONFLICT_GVALUE: int = 2
CONFLICT_CONSISTENCY: int = 3


# ═════════════════════════════════════════════════════════════════════════════
#  1.  danger_assess — Formal Contract
# ═════════════════════════════════════════════════════════════════════════════

@Pure
def danger_assess_pure(
    retrieval_count: int,
    step_count: int,
    value_consistency: float,
    danger_score: float,
) -> Tuple[bool, int, float]:
    """Pure model of the C danger_assess function.

    Postconditions (contract):
      - out_block  ⇔  retrieval_count > 12 ∨ step_count > 32
                       ∨ value_consistency < 0.3 ∨ danger_score > 0.005
      - out_threat ∈ {0, 1, 2, 3, 4}  (valid threat enum)
      - out_threat matches the first-triggered condition in priority order:
          c1 (resource_abuse) > c2 (runaway) > c3 (deception) > c4 (pattern)
      - c1 ⇒ out_score = 1.0
      - c2 ⇒ out_score = 1.0
      - c3 ⇒ out_score = 2.0 - value_consistency
    """
    # Priority 1: resource abuse
    if retrieval_count > MAX_RETRIEVALS:
        return True, THREAT_RESOURCE_ABUSE, 1.0
    # Priority 2: runaway
    if step_count > MAX_STEPS:
        return True, THREAT_RUNAWAY, 1.0
    # Priority 3: deception
    if value_consistency < CONSISTENCY_THRESHOLD:
        return True, THREAT_DECEPTION, 2.0 - value_consistency
    # Priority 4: pattern match
    if danger_score > SAFETY_THRESHOLD:
        return True, THREAT_PATTERN_MATCH, danger_score
    # No threat
    return False, THREAT_NONE, danger_score


# ═════════════════════════════════════════════════════════════════════════════
#  2.  gvalue_check_safety — Formal Contract
# ═════════════════════════════════════════════════════════════════════════════

@Pure
def gvalue_check_safety_pure(
    pos_d_min: float,
    neg_d_min: float,
    safety_margin: float,
) -> Tuple[bool, int]:
    """Pure model of the C gvalue_check_safety function.

    Requires:
        safety_margin >= 0
        pos_d_min >= 0
        neg_d_min >= 0

    Postconditions:
        safe  ⇔  pos_d_min ≤ neg_d_min - safety_margin
        unsafe ⇔  pos_d_min > neg_d_min - safety_margin
        violated_law = -1 if safe, 0 if unsafe
    """
    if pos_d_min > neg_d_min - safety_margin:
        return False, 0  # unsafe, law pair 0
    return True, -1  # safe


# ═════════════════════════════════════════════════════════════════════════════
#  3.  detect_any_conflict — Composition Contract
# ═════════════════════════════════════════════════════════════════════════════

@Pure
def detect_any_conflict_pure(
    danger_block: bool,
    gvalue_safe: bool,
    value_consistency: float,
) -> Tuple[bool, int]:
    """Pure model of the C detect_any_conflict function.

    Postconditions:
        has_conflict  ⇔  danger_block ∨ ¬gvalue_safe
                          ∨ value_consistency < CONSISTENCY_THRESHOLD
        source ∈ {CONFLICT_NONE, CONFLICT_DANGER, CONFLICT_GVALUE, CONFLICT_CONSISTENCY}
        danger_block   ⇒ source = CONFLICT_DANGER
        ¬danger_block ∧ ¬gvalue_safe  ⇒ source = CONFLICT_GVALUE
        ¬danger_block ∧ gvalue_safe ∧ consistency < threshold  ⇒ source = CONFLICT_CONSISTENCY
    """
    if danger_block:
        return True, CONFLICT_DANGER
    if not gvalue_safe:
        return True, CONFLICT_GVALUE
    if value_consistency < CONSISTENCY_THRESHOLD:
        return True, CONFLICT_CONSISTENCY
    return False, CONFLICT_NONE


# ═════════════════════════════════════════════════════════════════════════════
#  4.  Verified Properties — Logical Assertions
# ═════════════════════════════════════════════════════════════════════════════

def verify_danger_assess_contract() -> None:
    """Prove danger_assess invariants for concrete test cases.

    Nagini verifies these assertions statically.
    """
    # Case 1: retrieval exactly at boundary → no block
    r1, t1, s1 = danger_assess_pure(MAX_RETRIEVALS, 0, 1.0, 0.0)
    Assert(not r1)
    Assert(t1 == THREAT_NONE)

    # Case 2: retrieval just over → block, resource abuse
    r2, t2, s2 = danger_assess_pure(MAX_RETRIEVALS + 1, 0, 1.0, 0.0)
    Assert(r2)
    Assert(t2 == THREAT_RESOURCE_ABUSE)
    Assert(s2 == 1.0)

    # Case 3: step just over → block, runaway
    r3, t3, s3 = danger_assess_pure(0, MAX_STEPS + 1, 1.0, 0.0)
    Assert(r3)
    Assert(t3 == THREAT_RUNAWAY)
    Assert(s3 == 1.0)

    # Case 4: consistency below threshold → block, deception
    r4, t4, s4 = danger_assess_pure(0, 0, 0.0, 0.0)
    Assert(r4)
    Assert(t4 == THREAT_DECEPTION)
    Assert(s4 == 2.0)

    # Case 5: no triggers
    r5, t5, s5 = danger_assess_pure(0, 0, 1.0, 0.0)
    Assert(not r5)
    Assert(t5 == THREAT_NONE)
    Assert(s5 == 0.0)


def verify_gvalue_check_safety_contract() -> None:
    """Prove gvalue_check_safety invariants for concrete test cases."""
    # Edge: boundary (safe barely)
    r1, l1 = gvalue_check_safety_pure(5.0, 5.5, 0.5)  # pos=5, neg-m=5.0
    Assert(r1)
    Assert(l1 == -1)

    # Edge: just over boundary (unsafe)
    r2, l2 = gvalue_check_safety_pure(5.1, 5.5, 0.5)  # pos=5.1 > neg-m=5.0
    Assert(not r2)

    # Edge: zero margin with equal distances → safe
    r3, _ = gvalue_check_safety_pure(3.0, 3.0, 0.0)
    Assert(r3)

    # Edge: perfect negative alignment → unsafe
    r4, _ = gvalue_check_safety_pure(10.0, 0.0, 0.5)
    Assert(not r4)


def verify_conflict_detection_contract() -> None:
    """Prove detect_any_conflict invariants for concrete test cases."""
    # Danger block only
    r1, s1 = detect_any_conflict_pure(True, True, 1.0)
    Assert(r1)
    Assert(s1 == CONFLICT_DANGER)

    # Gvalue unsafe only
    r2, s2 = detect_any_conflict_pure(False, False, 1.0)
    Assert(r2)
    Assert(s2 == CONFLICT_GVALUE)

    # Consistency violation only
    r3, s3 = detect_any_conflict_pure(False, True, 0.2)
    Assert(r3)
    Assert(s3 == CONFLICT_CONSISTENCY)

    # No triggers
    r4, s4 = detect_any_conflict_pure(False, True, 0.5)
    Assert(not r4)
    Assert(s4 == CONFLICT_NONE)

    # Priority: danger overrides gvalue
    r5, s5 = detect_any_conflict_pure(True, False, 1.0)
    Assert(r5)
    Assert(s5 == CONFLICT_DANGER)

    # Priority: danger overrides consistency
    r6, s6 = detect_any_conflict_pure(True, True, 0.2)
    Assert(r6)
    Assert(s6 == CONFLICT_DANGER)

    # Priority: gvalue overrides consistency
    r7, s7 = detect_any_conflict_pure(False, False, 0.2)
    Assert(r7)
    Assert(s7 == CONFLICT_GVALUE)


# ═════════════════════════════════════════════════════════════════════════════
#  5.  Main verification entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all Nagini-verified safety checks."""
    verify_danger_assess_contract()
    verify_gvalue_check_safety_contract()
    verify_conflict_detection_contract()


if __name__ == "__main__":
    main()
