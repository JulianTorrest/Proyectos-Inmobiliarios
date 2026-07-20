from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormativeInputs:
    city: str
    area_m2: float
    floors_requested: int
    land_use: str


@dataclass(frozen=True)
class NormativeRule:
    city: str
    land_use: str
    max_floors: int
    max_height_m: float
    max_occupancy_ratio: float
    max_far: float  # floor area ratio
    notes: str


@dataclass(frozen=True)
class NormativeOutputs:
    allowed: bool
    reasons: list[str]
    max_floors: int
    max_height_m: float
    max_far: float
    max_occupancy_ratio: float


def evaluate_normative(inputs: NormativeInputs, rule: NormativeRule) -> NormativeOutputs:
    reasons: list[str] = []

    if inputs.city.strip().lower() != rule.city.strip().lower():
        reasons.append("La regla normativa no coincide con la ciudad seleccionada.")

    if inputs.land_use.strip().lower() != rule.land_use.strip().lower():
        reasons.append("El uso de suelo solicitado no coincide con la regla normativa seleccionada.")

    if inputs.floors_requested > rule.max_floors:
        reasons.append(
            f"Pisos solicitados ({inputs.floors_requested}) exceden el máximo permitido ({rule.max_floors})."
        )

    allowed = len(reasons) == 0
    return NormativeOutputs(
        allowed=allowed,
        reasons=reasons,
        max_floors=rule.max_floors,
        max_height_m=rule.max_height_m,
        max_far=rule.max_far,
        max_occupancy_ratio=rule.max_occupancy_ratio,
    )
