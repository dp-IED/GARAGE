"""Demo-only Pydantic schema for LM Studio structured output (injection fault types)."""

from typing import List, Literal

from pydantic import BaseModel, Field

# Must match keys in injection_catalog and dataset fault_types.
DemoFaultType = Literal[
    "normal",
    "COOLANT_DROPOUT",
    "VSS_DROPOUT",
    "MAF_SCALE_LOW",
    "TPS_STUCK",
    "gradual_drift",
]


class DemoFaultDiagnosis(BaseModel):
    faulty_sensors: List[str] = Field(default_factory=list)
    fault_type: DemoFaultType
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(
        ...,
        min_length=40,
        description=(
            "Required: 2–4 sentences in plain language. Cite which sensors exceed threshold in "
            "RAW GDN SCORES, any propagation/violations, and why fault_type fits (or why normal)."
        ),
    )
