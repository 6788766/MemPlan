"""
Placeholder definitions for TravelPlanner planlets.

These planlets correspond to the manually identified patterns in AGENT.md.  The
current implementation simply stores metadata; mining/matching will be layered
on top of this when the ComposeMatch stage is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


@dataclass(frozen=True)
class PlanletSpec:
    """
    Describes a parameterised subgraph (planlet) in terms of a pattern identifier
    and the set of (City, Date) binders it covers.
    """

    planlet_id: str
    description: str
    required_slots: Sequence[str]


PLANLETS: Dict[str, PlanletSpec] = {
    "ArrivalDate": PlanletSpec(
        planlet_id="ArrivalDate",
        description="Inbound move + stay + first-day meals/visits for a city.",
        required_slots=("Move", "Stay", "Eat", "Visit"),
    ),
    "CityDate": PlanletSpec(
        planlet_id="CityDate",
        description="Full day in a city (stay, meals, visit).",
        required_slots=("Stay", "Eat", "Visit"),
    ),
    "TwoEatsOneVisit": PlanletSpec(
        planlet_id="TwoEatsOneVisit",
        description="Subset focusing on meal/attraction diversity constraints.",
        required_slots=("Eat", "Visit"),
    ),
    "DepartureDate": PlanletSpec(
        planlet_id="DepartureDate",
        description="Last-day stay + outbound move.",
        required_slots=("Stay", "Move"),
    ),
}

