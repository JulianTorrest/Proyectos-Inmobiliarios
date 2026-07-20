from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return project_root() / "data"


def generate_parametros() -> pd.DataFrame:
    cities = [
        ("Bogotá", 9_500_000, 3_600_000, 0.18),
        ("Medellín", 8_500_000, 3_300_000, 0.17),
        ("Cali", 6_500_000, 3_000_000, 0.17),
        ("Cartagena", 7_800_000, 3_400_000, 0.18),
        ("Barranquilla", 7_000_000, 3_100_000, 0.17),
        ("Bucaramanga", 6_200_000, 2_900_000, 0.17),
        ("Pereira", 5_800_000, 2_800_000, 0.17),
        ("Manizales", 5_500_000, 2_750_000, 0.16),
        ("Ibagué", 4_800_000, 2_500_000, 0.16),
        ("Santa Marta", 6_800_000, 3_100_000, 0.17),
        ("Cúcuta", 4_300_000, 2_400_000, 0.16),
        ("Villavicencio", 4_600_000, 2_450_000, 0.16),
        ("Pasto", 4_500_000, 2_400_000, 0.16),
        ("Montería", 4_700_000, 2_500_000, 0.16),
        ("Valledupar", 4_400_000, 2_400_000, 0.16),
        ("Armenia", 5_200_000, 2_650_000, 0.16),
        ("Neiva", 4_500_000, 2_450_000, 0.16),
        ("Popayán", 4_300_000, 2_380_000, 0.16),
        ("Sincelejo", 4_200_000, 2_350_000, 0.16),
        ("Tunja", 4_600_000, 2_450_000, 0.16),
        ("Riohacha", 4_100_000, 2_300_000, 0.16),
        ("Florencia", 4_000_000, 2_300_000, 0.16),
        ("Yopal", 4_500_000, 2_420_000, 0.16),
        ("Quibdó", 3_800_000, 2_250_000, 0.16),
        ("Buenaventura", 4_200_000, 2_380_000, 0.16),
    ]

    uses = {
        "Residencial": {"price_factor": 1.00, "cost_factor": 1.00, "soft_add": 0.00, "size_add": 0, "sales_add": 0, "build_add": 0},
        "Mixto": {"price_factor": 1.10, "cost_factor": 1.10, "soft_add": 0.01, "size_add": 5, "sales_add": 2, "build_add": 2},
        "Comercial": {"price_factor": 1.20, "cost_factor": 1.15, "soft_add": 0.03, "size_add": 35, "sales_add": 6, "build_add": 4},
        "Oficinas": {"price_factor": 1.10, "cost_factor": 1.12, "soft_add": 0.02, "size_add": 20, "sales_add": 4, "build_add": 4},
    }

    rows = []
    for city, base_price, base_cost, rate in cities:
        for use, f in uses.items():
            rows.append({
                "city": city,
                "land_use": use,
                "price_per_m2_sell": int(round(base_price * f["price_factor"])),
                "cost_per_m2_build": int(round(base_cost * f["cost_factor"])),
                "soft_cost_pct": 0.13 + f["soft_add"],
                "avg_unit_size_m2": 60 + f["size_add"],
                "sales_months": 16 + f["sales_add"],
                "build_months": 18 + f["build_add"],
                "discount_rate_annual": rate,
            })
    return pd.DataFrame(rows)


def generate_normativa(parametros: pd.DataFrame) -> pd.DataFrame:
    def rule_for(city: str, use: str) -> dict:
        tier = {
            "Bogotá": 5, "Medellín": 5, "Cali": 4, "Cartagena": 4, "Barranquilla": 4,
            "Bucaramanga": 3, "Pereira": 3, "Manizales": 3, "Ibagué": 3,
            "Santa Marta": 3, "Cúcuta": 3, "Villavicencio": 3, "Pasto": 3,
            "Montería": 3, "Valledupar": 2, "Armenia": 3, "Neiva": 3,
            "Popayán": 2, "Sincelejo": 2, "Tunja": 3, "Riohacha": 2,
            "Florencia": 2, "Yopal": 2, "Quibdó": 2, "Buenaventura": 2,
        }.get(city, 2)

        base_floors = {"Residencial": 8, "Mixto": 12, "Comercial": 6, "Oficinas": 14}
        base_far = {"Residencial": 3.0, "Mixto": 4.2, "Comercial": 3.2, "Oficinas": 4.8}

        floors = int(base_floors[use] + tier * 1.2)
        far = round(base_far[use] + tier * 0.4, 1)
        height = int(floors * 3.5)
        occ = 0.55 if use in ("Residencial", "Oficinas") else 0.65

        return {
            "city": city,
            "land_use": use,
            "max_floors": floors,
            "max_height_m": height,
            "max_occupancy_ratio": occ,
            "max_far": far,
            "notes": "Dummy: valores referenciales, NO oficiales",
        }

    rows = []
    for _, r in parametros.iterrows():
        rows.append(rule_for(r["city"], r["land_use"]))
    return pd.DataFrame(rows)


def generate_cronograma(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range(start="2026-01-02", end="2026-12-31", periods=n)
    weight = round(1.0 / n, 6)

    phase_templates = [
        "Actas y permisos de {i}",
        "Cerramiento perimetral - sección {i}",
        "Excavación y descapote zona {i}",
        "Cimentación zapata {i}",
        "Estructura columna placa piso {i}",
        "Mampostería divisoria piso {i}",
        "Instalaciones eléctricas piso {i}",
        "Instalaciones hidrosanitarias piso {i}",
        "Acabados enchapes piso {i}",
        "Acabados pintura piso {i}",
        "Pruebas y ajustes piso {i}",
        "Entrega documental lote {i}",
    ]

    milestones = []
    for i in range(n):
        template = phase_templates[i % len(phase_templates)]
        milestones.append(template.format(i=i + 1))

    df = pd.DataFrame({
        "milestone": milestones,
        "planned_date": [d.date().isoformat() for d in dates],
        "weight": weight,
    })
    # Force exact sum to 1.0
    df.at[0, "weight"] = weight + (1.0 - weight * n)
    return df


def main() -> None:
    data_dir().mkdir(parents=True, exist_ok=True)

    parametros = generate_parametros()
    normativa = generate_normativa(parametros)
    cronograma = generate_cronograma(100)

    parametros.to_csv(data_dir() / "parametros_ciudades.csv", index=False)
    normativa.to_csv(data_dir() / "normativa_ciudades.csv", index=False)
    cronograma.to_csv(data_dir() / "cronograma_planeado.csv", index=False)

    print(f"Parametros: {len(parametros)} rows")
    print(f"Normativa: {len(normativa)} rows")
    print(f"Cronograma: {len(cronograma)} rows")


if __name__ == "__main__":
    main()
