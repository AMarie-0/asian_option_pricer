"""
registry.py — Central catalog of priceable options.

Each option is described by an OptionSpec:
  - which pricing engine it needs ("lattice" = O(n^2), "paths" = O(2^n))
  - which extra inputs the UI must render (strike, barrier, ...)
  - a pricing function with the uniform signature f(params, **inputs) -> dict

app.py can then build its option dropdown and input widgets generically:

    from src.options.registry import REGISTRY
    spec   = REGISTRY[selected_key]
    result = spec.price(params, **user_inputs)

Adding option #11 later = one function + one register() call. No UI surgery.

Engine note: "paths" options inherit the 2^n cost of full enumeration, so
they must respect the same MAX_N cap as the existing Asian pricer (25 local,
20 on Streamlit Cloud). "lattice" options are O(n^2) and can safely use
hundreds of steps — the registry records a recommended default per engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.model.calibration import ModelParams

# Recommended step counts per engine (UI defaults, not hard limits)
DEFAULT_N = {"paths": 25, "lattice": 200}


@dataclass(frozen=True)
class ExtraInput:
    """One extra widget the UI must render for this option."""
    key:     str                 # kwarg name passed to the pricer
    label:   str                 # widget label
    kind:    str = "moneyness"   # "moneyness" (% of S0) | "float" | "choice"
    default: float = 1.0
    minimum: float = 0.01
    maximum: float = 5.0
    help:    str = ""
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptionSpec:
    key:         str
    name:        str
    category:    str                    # "classic" | "exotic"
    engine:      str                    # "lattice" | "paths"
    price:       Callable[..., dict]    # f(params: ModelParams, **inputs)
    description: str
    extra_inputs: tuple[ExtraInput, ...] = field(default_factory=tuple)

    @property
    def default_n(self) -> int:
        return DEFAULT_N[self.engine]


REGISTRY: dict[str, OptionSpec] = {}


def register(spec: OptionSpec) -> None:
    if spec.key in REGISTRY:
        raise ValueError(f"Duplicate option key: {spec.key}")
    REGISTRY[spec.key] = spec


def by_category(category: str) -> list[OptionSpec]:
    return [s for s in REGISTRY.values() if s.category == category]


# Importing the option modules populates the registry.
from src.options import classic as _classic   # noqa: E402,F401
from src.options import exotic as _exotic     # noqa: E402,F401
