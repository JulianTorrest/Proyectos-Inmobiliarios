from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FinanceInputs:
    total_units: int
    sell_price_per_unit: float
    construction_cost_total: float
    soft_costs_total: float
    land_cost: float
    sales_months: int
    build_months: int
    discount_rate_annual: float
    tax_rate: float = 0.0
    transaction_cost_pct: float = 0.0


@dataclass(frozen=True)
class FinanceOutputs:
    revenue_total: float
    costs_total: float
    profit_total: float
    profit_margin: float
    npv: float
    irr_annual: float | None


def _irr_monthly(cashflows: np.ndarray) -> float | None:
    if cashflows.size < 2:
        return None

    if not (np.any(cashflows > 0) and np.any(cashflows < 0)):
        return None

    # Solve: cf0 + cf1*x + ... + cfn*x^n = 0, where x = 1/(1+r)
    coeffs = cashflows.astype(float)
    try:
        roots = np.roots(coeffs[::-1])
    except Exception:
        return None

    real_roots = roots[np.isclose(roots.imag, 0.0)].real
    real_roots = real_roots[real_roots > 0]
    if real_roots.size == 0:
        return None

    rates = (1.0 / real_roots) - 1.0
    rates = rates[np.isfinite(rates)]
    if rates.size == 0:
        return None

    # Choose a "reasonable" rate: closest to 0 but not below -99%
    rates = rates[rates > -0.99]
    if rates.size == 0:
        return None

    return float(rates[np.argmin(np.abs(rates))])


def _monthly_rate(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def evaluate_project(inputs: FinanceInputs) -> FinanceOutputs:
    revenue_gross = float(inputs.total_units * inputs.sell_price_per_unit)
    revenue_total = revenue_gross * (1.0 - inputs.transaction_cost_pct)
    costs_total = float(inputs.construction_cost_total + inputs.soft_costs_total + inputs.land_cost)
    profit_before_tax = revenue_total - costs_total
    tax = max(0.0, profit_before_tax * inputs.tax_rate)
    profit_total = profit_before_tax - tax
    profit_margin = profit_total / revenue_gross if revenue_gross > 0 else 0.0

    m_rate = _monthly_rate(inputs.discount_rate_annual)

    months = max(int(inputs.build_months) + int(inputs.sales_months), 1)

    cashflows = np.zeros(months + 1, dtype=float)
    cashflows[0] = -inputs.land_cost

    build_months = max(int(inputs.build_months), 1)
    sales_months = max(int(inputs.sales_months), 1)

    build_outflow = (inputs.construction_cost_total + inputs.soft_costs_total) / build_months
    for t in range(1, build_months + 1):
        cashflows[t] -= build_outflow

    sales_inflow = revenue_total / sales_months
    for i in range(1, sales_months + 1):
        t = build_months + i
        if t <= months:
            cashflows[t] += sales_inflow

    if months >= 1:
        cashflows[months] -= tax

    npv = float(sum(cashflows[t] / ((1.0 + m_rate) ** t) for t in range(0, months + 1)))

    irr_annual = None
    irr_monthly = _irr_monthly(cashflows)
    if irr_monthly is not None and np.isfinite(irr_monthly):
        irr_annual = (1.0 + irr_monthly) ** 12 - 1.0

    return FinanceOutputs(
        revenue_total=revenue_total,
        costs_total=costs_total,
        profit_total=profit_total,
        profit_margin=float(profit_margin),
        npv=npv,
        irr_annual=irr_annual,
    )


def compute_debt_schedule(
    principal: float,
    annual_rate: float,
    term_years: int,
    grace_years: int = 0,
) -> list[dict]:
    months = int(term_years * 12)
    grace_months = int(grace_years * 12)
    monthly_rate = _monthly_rate(annual_rate)
    rows: list[dict] = []
    balance = principal
    for m in range(1, months + 1):
        interest = balance * monthly_rate
        disbursement = principal if m == 1 else 0.0
        if m <= grace_months:
            principal_payment = 0.0
            payment = interest
        else:
            remaining = months - m + 1
            if monthly_rate == 0.0 or remaining <= 0:
                principal_payment = balance / max(remaining, 1)
                payment = principal_payment + interest
            else:
                payment = balance * (monthly_rate * (1.0 + monthly_rate) ** remaining) / ((1.0 + monthly_rate) ** remaining - 1.0)
                principal_payment = payment - interest
        balance -= principal_payment
        rows.append(
            {
                "periodo": m,
                "desembolso": disbursement,
                "interes": interest,
                "capital": principal_payment,
                "cuota": payment,
                "saldo": max(0.0, balance),
            }
        )
    return rows
