"""
TravelPlanner-specific adapters for the MemPlan pipeline.

The goal is to isolate all benchmark-specific hooks (tool wrappers, cost helpers,
etc.) so the core algorithms remain domain agnostic.  At this stage we only
expose light wrappers that will be extended alongside the algorithm
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from task_helper.travel.template import (
    TravelPlanTemplate,
    build_travel_template,
    render_template_prompt,
)
from memory_graph import fetch as memory_fetch
from memory_graph.schema import MemoryGraph
from task_helper.travel.utils.paths import travel_dataset_root


def _strip_parenthetical(city: str) -> str:
    """
    Normalise TravelPlanner city strings like ``'Buffalo(New York)'`` to just
    the city name expected by the tool CSVs.
    """

    if not city:
        return city
    idx = city.find("(")
    if idx == -1:
        return city.strip()
    return city[:idx].strip()

@dataclass
class TravelToolset:
    """
    Lazily-instantiated wrapper around the TravelPlanner tool APIs.
    """

    flights: Any
    accommodations: Any
    restaurants: Any
    attractions: Any
    distance: Any

    @classmethod
    def load(cls) -> "TravelToolset":
        dataset_root = travel_dataset_root()
        from task_helper.travel.tools.flights.apis import Flights
        from task_helper.travel.tools.accommodations.apis import Accommodations
        from task_helper.travel.tools.restaurants.apis import Restaurants
        from task_helper.travel.tools.attractions.apis import Attractions
        from task_helper.travel.tools.googleDistanceMatrix.apis import GoogleDistanceMatrix

        flights = Flights(path=str(dataset_root / "database/flights/clean_Flights_2022.csv"))
        accommodations = Accommodations(
            path=str(dataset_root / "database/accommodations/clean_accommodations_2022.csv")
        )
        restaurants = Restaurants(
            path=str(dataset_root / "database/restaurants/clean_restaurant_2022.csv")
        )
        attractions = Attractions(
            path=str(dataset_root / "database/attractions/attractions.csv")
        )
        distance = GoogleDistanceMatrix(data_path=str(dataset_root / "database/googleDistanceMatrix/distance.csv"))

        return cls(
            flights=flights,
            accommodations=accommodations,
            restaurants=restaurants,
            attractions=attractions,
            distance=distance,
        )


class Fetch:
    """
    Adapter used by ComposeMatch/PlanletCover (once implemented) to retrieve
    either memory-backed planlets or fresh tool results.
    """

    def __init__(self, tools: TravelToolset) -> None:
        self.tools = tools
        self._distance_cache: Dict[Tuple[str, str, str], dict] = {}

    # ------------------------------------------------------------------
    # Memory-backed views
    # ------------------------------------------------------------------
    def planlet(
        self, planlet_id: str, scopes: Sequence[Tuple[str, str]], graph: MemoryGraph
    ):
        yield from memory_fetch.planlet_embeddings(planlet_id, scopes, graph)

    # ------------------------------------------------------------------
    # Tool fetchers (basic implementation)
    # ------------------------------------------------------------------
    def flights(self, origin: str, destination: str, date: str):
        origin_norm = _strip_parenthetical(origin)
        destination_norm = _strip_parenthetical(destination)
        return self.tools.flights.run(origin_norm, destination_norm, date)

    def accommodations(self, city: str):
        city_norm = _strip_parenthetical(city)
        result = self.tools.accommodations.run(city_norm)
        if isinstance(result, str) or getattr(result, "empty", False):
            run_alt = getattr(self.tools.accommodations, "run_for_annotation", None)
            if callable(run_alt):
                result = run_alt(city_norm)
        return result

    def restaurants(self, city: str):
        city_norm = _strip_parenthetical(city)
        result = self.tools.restaurants.run(city_norm)
        if isinstance(result, str) or getattr(result, "empty", False):
            run_alt = getattr(self.tools.restaurants, "run_for_annotation", None)
            if callable(run_alt):
                result = run_alt(city_norm)
        return result

    def attractions(self, city: str):
        city_norm = _strip_parenthetical(city)
        result = self.tools.attractions.run(city_norm)
        if isinstance(result, str) or getattr(result, "empty", False):
            run_alt = getattr(self.tools.attractions, "run_for_annotation", None)
            if callable(run_alt):
                result = run_alt(city_norm)
        return result

    def distance(self, origin: str, destination: str, mode: str = "driving"):
        origin_norm = _strip_parenthetical(origin)
        destination_norm = _strip_parenthetical(destination)
        key = (origin_norm, destination_norm, mode)
        if key not in self._distance_cache:
            self._distance_cache[key] = self.tools.distance.run_for_evaluation(
                origin_norm, destination_norm, mode=mode
            )
        return self._distance_cache[key]


def create_fetch() -> Fetch:
    tools = TravelToolset.load()
    return Fetch(tools)


__all__ = [
    "TravelToolset",
    "Fetch",
    "create_fetch",
    "TravelPlanTemplate",
    "build_travel_template",
    "render_template_prompt",
]
