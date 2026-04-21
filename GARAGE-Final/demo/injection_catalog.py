"""
Fault injection type labels and short descriptions for LLM prompts (demo-only).

Aligned with strings produced by data injection (see data/create_shared_dataset.py).
"""

# Keys must match fault_types strings in the shared dataset (plus normal).
INJECTION_FAULT_DESCRIPTIONS: dict[str, str] = {
    "normal": "No synthetic fault; vehicle operating within expected norms.",
    "COOLANT_DROPOUT": (
        "Coolant temperature signal dropped or flatlined unexpectedly, often affecting "
        "related thermal and fuel-trim behavior."
    ),
    "VSS_DROPOUT": (
        "Vehicle speed sensor signal lost or implausible while other drivetrain signals "
        "still look active — breaks expected RPM/speed correlation."
    ),
    "MAF_SCALE_LOW": (
        "Mass-air / intake-related readings skew low versus throttle and load, suggesting "
        "airflow scaling or manifold pressure inconsistency."
    ),
    "TPS_STUCK": (
        "Throttle position appears stuck or nearly constant despite changing engine load "
        "or driver demand."
    ),
    "gradual_drift": (
        "Slow drift across one or more correlated sensors rather than a sharp dropout."
    ),
}


def format_injection_block_for_prompt() -> str:
    """Build the user-message suffix listing allowed fault_type values with descriptions."""
    lines = [
        "FAULT TYPE (choose exactly one value for JSON field fault_type):",
        "",
    ]
    for key in ("normal", "COOLANT_DROPOUT", "VSS_DROPOUT", "MAF_SCALE_LOW", "TPS_STUCK", "gradual_drift"):
        desc = INJECTION_FAULT_DESCRIPTIONS.get(key, "")
        lines.append(f"- {key}: {desc}")
    lines.append("")
    lines.append(
        "If evidence supports normal operation, choose normal. Otherwise pick the best "
        "matching injected fault label from the list above."
    )
    return "\n".join(lines)
