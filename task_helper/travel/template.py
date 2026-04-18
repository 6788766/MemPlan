"""
Utilities for constructing the structured query template (``Q0``) that the
MemPlan pipeline consumes before invoking memory-backed planning.

The template captures:
    * canonical city/date placeholders derived from the NL query;
    * an action skeleton expressing which activities must be filled;
    * hard constraints (budget, transport bans, room types, etc.);
    * coverage preferences (e.g. cuisine diversity).
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    openai = None
else:
    # Best-effort: load OPENAI_API_KEY from .env when not provided.
    if not (openai.api_key or os.getenv("OPENAI_API_KEY")):
        env_path = Path(".env")
        if env_path.exists():
            with env_path.open(encoding="utf-8") as fp:
                for line in fp:
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        if key:
                            os.environ.setdefault("OPENAI_API_KEY", key)
                        break


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HardConstraints:
    budget: Optional[float]
    people: Optional[int]
    transport_restriction: Optional[str]
    room_type: Optional[str]
    house_rule: Optional[str]
    level: Optional[str] = None
    visiting_city_number: Optional[int] = None


@dataclass
class CoveragePreferences:
    cuisine: Sequence[str] = field(default_factory=tuple)


@dataclass
class ActionPlaceholder:
    day: int
    action_type: str
    description: str
    city_slot: str
    required: bool = True
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "day": self.day,
            "action_type": self.action_type,
            "description": self.description,
            "city_slot": self.city_slot,
            "required": self.required,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass
class TravelPlanTemplate:
    origin: str
    destination: str
    destination_kind: str  # "city" or "state"
    days: int
    dates: Sequence[str]
    city_slots: Sequence[str]
    city_schedule: Sequence[str]
    actions: Sequence[ActionPlaceholder]
    hard_constraints: HardConstraints
    coverage_preferences: CoveragePreferences
    notes: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "destination_kind": self.destination_kind,
            "days": self.days,
            "dates": list(self.dates),
            "city_slots": list(self.city_slots),
            "city_schedule": list(self.city_schedule),
            "actions": [action.to_dict() for action in self.actions],
            "hard_constraints": vars(self.hard_constraints),
            "coverage_preferences": {
                "cuisine": list(self.coverage_preferences.cuisine)
            },
            "notes": dict(self.notes),
        }


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------


def build_travel_template(nl_query: Dict[str, object]) -> TravelPlanTemplate:
    """
    Build the ``Q0`` skeleton from a TravelPlanner query dict.

    Args:
        nl_query: record from the TravelPlanner dataset (train/validation/test). Only the
            natural language query text is used; structured fields are inferred via LLM.

    Returns:
        A populated :class:`TravelPlanTemplate`.
    """

    query_text = str(nl_query.get("query", "")).strip()
    if not query_text:
        raise ValueError("Natural language query text is required.")

    llm_used = False
    llm_error: Optional[str] = None

    # Toggle: default LLM; if ALLOW_RULE_BASE=true and LLM fails/unavailable, fall back.
    allow_rule_base = os.getenv("ALLOW_RULE_BASE", "false").lower() in {"1", "true", "yes"}
    use_llm = (
        openai is not None
        and (openai.api_key or os.getenv("OPENAI_API_KEY"))
    )
    if use_llm:
        try:
            facts = _extract_facts_with_llm(query_text)
            llm_used = True
        except Exception as exc:
            llm_error = f"LLM extraction failed: {exc}"
            if not allow_rule_base:
                raise
            facts = _extract_facts_fallback(nl_query)
    else:
        if openai is None:
            llm_error = "openai package not installed; using rule-based parser."
        elif not (openai.api_key or os.getenv("OPENAI_API_KEY")):
            llm_error = "OPENAI_API_KEY not set; using rule-based parser."
        else:
            llm_error = "LLM disabled; using rule-based parser."
        if not allow_rule_base:
            raise RuntimeError(llm_error)
        facts = _extract_facts_fallback(nl_query)

    origin = facts.get("origin_city") or "UNKNOWN ORIGIN"
    dest_info = facts.get("destination") or {}
    dest_raw = dest_info.get("name") or "UNKNOWN DESTINATION"
    destination_kind = (dest_info.get("type") or "").lower()
    if destination_kind not in {"city", "state"}:
        destination_kind = "state" if dest_info.get("multi_city", False) else "city"

    dates = []
    raw_dates = facts.get("dates") or []
    if isinstance(raw_dates, list):
        dates = [str(item) for item in raw_dates if item]
    elif isinstance(raw_dates, str) and raw_dates:
        dates = [raw_dates]

    days = facts.get("days")
    if days is None:
        days = len(dates) if dates else 3
    days = max(int(days), 1)

    visiting_city_number = facts.get("visiting_city_number")
    if visiting_city_number is None:
        visiting_city_number = 2 if destination_kind == "state" else 1
    visiting_city_number = max(int(visiting_city_number), 1)

    city_slots = _derive_city_slots(dest_raw, destination_kind, visiting_city_number)
    city_schedule = _assign_city_schedule(days, city_slots)

    actions = _build_action_skeleton(
        days=days,
        origin=origin,
        city_schedule=city_schedule,
        city_slots=city_slots,
    )

    hard_constraints = _build_hard_constraints_from_facts(facts)
    coverage_preferences = _build_coverage_preferences_from_facts(facts)

    notes: Dict[str, object] = {
        "query_text": query_text,
        "budget_currency": "USD",
    }
    notes["llm_source"] = "openai" if llm_used else "fallback"
    if llm_used:
        notes["llm_model"] = LLM_MODEL
    if llm_error:
        notes["llm_error"] = llm_error
    if facts.get("notes"):
        notes["llm_notes"] = facts["notes"]
    if destination_kind == "state":
        notes["destination_hint"] = (
            "Assign concrete cities inside the destination state before planning."
        )

    return TravelPlanTemplate(
        origin=origin,
        destination=dest_raw,
        destination_kind=destination_kind,
        days=days,
        dates=dates,
        city_slots=tuple(city_slots),
        city_schedule=tuple(city_schedule),
        actions=tuple(actions),
        hard_constraints=hard_constraints,
        coverage_preferences=coverage_preferences,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_city_slots(
    destination: str, destination_kind: str, visiting_city_number: int
) -> List[str]:
    if destination_kind == "city":
        return [destination or "DEST_CITY"]

    count = max(visiting_city_number, 2) if destination_kind == "state" else 1
    return [f"CITY_{idx+1}" for idx in range(count)]


def _assign_city_schedule(days: int, city_slots: Sequence[str]) -> List[str]:
    if not days:
        return []
    if not city_slots:
        city_slots = ("CITY_1",)

    segments = len(city_slots)
    base = days // segments
    remainder = days % segments

    schedule: List[str] = []
    day_pointer = 0
    for idx, slot in enumerate(city_slots):
        span = base + (1 if idx < remainder else 0)
        if span == 0:
            span = 1  # ensure coverage
        for _ in range(span):
            if day_pointer < days:
                schedule.append(slot)
                day_pointer += 1

    # If rounding produced fewer entries than days, pad with last slot.
    while len(schedule) < days:
        schedule.append(schedule[-1])
    return schedule


def _build_action_skeleton(
    days: int,
    origin: str,
    city_schedule: Sequence[str],
    city_slots: Sequence[str],
) -> List[ActionPlaceholder]:
    actions: List[ActionPlaceholder] = []
    if days == 0:
        return actions

    # Day 1 inbound move.
    first_city = city_schedule[0]
    actions.append(
        ActionPlaceholder(
            day=1,
            action_type="MoveIn",
            description=f"Travel from {origin} to {first_city}",
            city_slot=first_city,
            notes="Select transportation mode compliant with constraints.",
        )
    )

    for day_idx, city in enumerate(city_schedule, start=1):
        actions.append(
            ActionPlaceholder(
                day=day_idx,
                action_type="Stay",
                description=f"Book accommodation in {city}",
                city_slot=city,
                required=day_idx <= max(days - 1, 2),
            )
        )
        actions.append(
            ActionPlaceholder(
                day=day_idx,
                action_type="Breakfast",
                description=f"Breakfast in {city}",
                city_slot=city,
            )
        )
        actions.append(
            ActionPlaceholder(
                day=day_idx,
                action_type="Lunch",
                description=f"Lunch in {city}",
                city_slot=city,
            )
        )
        actions.append(
            ActionPlaceholder(
                day=day_idx,
                action_type="Dinner",
                description=f"Dinner in {city}",
                city_slot=city,
            )
        )
        actions.append(
            ActionPlaceholder(
                day=day_idx,
                action_type="Visit",
                description=f"Primary attractions in {city}",
                city_slot=city,
            )
        )

        # Inter-city move between consecutive days if the city changes.
        if day_idx < days:
            next_city = city_schedule[day_idx]
            if next_city != city:
                actions.append(
                    ActionPlaceholder(
                        day=day_idx + 1,
                        action_type="Move",
                        description=f"Travel from {city} to {next_city}",
                        city_slot=next_city,
                    )
                )

    # Outbound move back to origin on the final day.
    last_city = city_schedule[-1]
    actions.append(
        ActionPlaceholder(
            day=days,
            action_type="MoveOut",
            description=f"Return from {last_city} to {origin}",
            city_slot=last_city,
            notes="Can be same-day evening flight or final-day transport.",
        )
    )
    return actions


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_template_prompt(template: TravelPlanTemplate) -> str:
    """
    Construct a planner-friendly instruction string describing the template.
    """

    lines: List[str] = []
    lines.append("You are generating a travel plan skeleton based on validated memories.")
    lines.append(
        f"Trip summary: {template.days} day(s) for {template.hard_constraints.people} "
        f"traveller(s) from {template.origin} to {template.destination} ({template.destination_kind})."
    )
    if template.dates:
        lines.append(f"Dates: {', '.join(template.dates)}")
    lines.append("")
    lines.append("City slots (assign concrete cities later if needed):")
    for slot in template.city_slots:
        lines.append(f"  - {slot}")
    lines.append("")

    lines.append("Hard constraints to respect:")
    hc = template.hard_constraints
    lines.append(f"  - Budget (USD): {hc.budget if hc.budget is not None else 'unspecified'}")
    lines.append(f"  - People: {hc.people if hc.people is not None else 'unspecified'}")
    lines.append(
        f"  - Transport restriction: {hc.transport_restriction or 'none (default: allowed)'}"
    )
    lines.append(f"  - Room type preference: {hc.room_type or 'none'}")
    lines.append(f"  - House rule preference: {hc.house_rule or 'none'}")
    if template.coverage_preferences.cuisine:
        lines.append(
            "  - Cuisine coverage: include "
            + ", ".join(template.coverage_preferences.cuisine)
        )
    lines.append("")

    lines.append("Action skeleton (fill placeholders with validated memories):")
    for action in template.actions:
        note_suffix = f" [{action.notes}]" if action.notes else ""
        required = "required" if action.required else "optional"
        lines.append(
            f"  - Day {action.day:>2}: {action.action_type} in {action.city_slot} "
            f"({required}) -> {action.description}{note_suffix}"
        )
    lines.append("")
    lines.append(
        "Use the memory graph to source stays, meals, attractions, and transportation "
        "that match the city slots and constraints. For any placeholder lacking memory "
        "coverage, fall back to the live tools."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


LLM_MODEL = os.getenv("MEMPLAN_LLM_MODEL", "gpt-5-mini")
LLM_SYSTEM_PROMPT = """You convert natural language travel requests into structured data.
Return a JSON object with the following schema (omit no keys, use null for unknown values):
{
  "origin_city": <string|null>,
  "destination": {
    "name": <string|null>,
    "type": "city"|"state"|null,
    "multi_city": <bool|null>
  },
  "dates": [<ISO date strings>?],
  "days": <integer or null>,
  "people": <integer or null>,
  "budget": <number or null>,
  "visiting_city_number": <integer or null>,
  "constraints": {
    "transportation": <string|null>,  # e.g., "no flight"
    "room_type": <string|null>,
    "house_rule": <string|null>,
    "cuisine": [<string>]
  },
  "notes": <string|null>
}
Always output valid JSON only."""


def _extract_facts_with_llm(query_text: str) -> Dict[str, object]:
    api_key = getattr(openai, "api_key", None) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required for LLM-based template generation."
        )
    if openai is None:
        raise RuntimeError("openai package not available")
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": query_text},
    ]
    # Prefer new 1.x client; fall back to legacy if present.
    if hasattr(openai, "Client"):
        client = openai.Client(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
        )
        content = response.choices[0].message.content
    elif hasattr(openai, "ChatCompletion"):
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
        )
        content = response["choices"][0]["message"]["content"]
    else:
        raise RuntimeError("Unsupported openai client version")
    json_text = _extract_json_block(content)
    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object.")
    return data


def _extract_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM did not return JSON content.")
    return match.group(0)



def _extract_facts_fallback(nl_query: Dict[str, object]) -> Dict[str, object]:
    origin = str(nl_query.get("org", "") or "").strip() or None
    destination_raw = str(nl_query.get("dest", "") or "").strip() or None
    visiting_city_number = nl_query.get("visiting_city_number")
    try:
        visiting_city_number = int(visiting_city_number) if visiting_city_number is not None else None
    except (TypeError, ValueError):
        visiting_city_number = None
    if visiting_city_number is None:
        visiting_city_number = 1
    destination_type = "state" if visiting_city_number > 1 else "city"
    dates = nl_query.get("date") or []
    if isinstance(dates, str):
        try:
            dates = ast.literal_eval(dates)
        except (SyntaxError, ValueError):
            dates = []
    if not isinstance(dates, list):
        dates = list(dates)
    days = nl_query.get("days")
    try:
        days = int(days) if days is not None else None
    except (TypeError, ValueError):
        days = None
    if days is None:
        days = len(dates) if dates else 3
    people = nl_query.get("people_number")
    try:
        people = int(people) if people is not None else None
    except (TypeError, ValueError):
        people = None
    budget = nl_query.get("budget")
    try:
        budget = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None
    local_constraint = nl_query.get("local_constraint") or {}
    if isinstance(local_constraint, str):
        try:
            local_constraint = ast.literal_eval(local_constraint)
        except (SyntaxError, ValueError):
            local_constraint = {}
    transport = local_constraint.get("transportation")
    room_type = local_constraint.get("room type")
    house_rule = local_constraint.get("house rule")
    cuisine = local_constraint.get("cuisine")
    if isinstance(cuisine, str):
        cuisine = [item.strip() for item in cuisine.split(",") if item.strip()]
    elif not cuisine:
        cuisine = []
    facts = {
        "origin_city": origin,
        "destination": {
            "name": destination_raw,
            "type": destination_type,
            "multi_city": visiting_city_number > 1,
        },
        "dates": dates,
        "days": days,
        "people": people,
        "budget": budget,
        "visiting_city_number": visiting_city_number,
        "constraints": {
            "transportation": transport,
            "room_type": room_type,
            "house_rule": house_rule,
            "cuisine": cuisine,
        },
        "notes": "fallback_from_structured_fields",
    }
    return facts


def _build_hard_constraints_from_facts(facts: Dict[str, object]) -> HardConstraints:
    constraints = facts.get("constraints") or {}
    budget = facts.get("budget")
    try:
        budget = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None

    people = facts.get("people")
    try:
        people = int(people) if people is not None else None
    except (TypeError, ValueError):
        people = None

    transport = constraints.get("transportation")
    room_type = constraints.get("room_type")
    house_rule = constraints.get("house_rule")

    visiting_city_number = facts.get("visiting_city_number")
    try:
        visiting_city_number = (
            int(visiting_city_number) if visiting_city_number is not None else None
        )
    except (TypeError, ValueError):
        visiting_city_number = None

    return HardConstraints(
        budget=budget,
        people=people,
        transport_restriction=transport,
        room_type=room_type,
        house_rule=house_rule,
        level=None,
        visiting_city_number=visiting_city_number,
    )


def _build_coverage_preferences_from_facts(
    facts: Dict[str, object]
) -> CoveragePreferences:
    constraints = facts.get("constraints") or {}
    cuisine = constraints.get("cuisine") or []
    if isinstance(cuisine, str):
        cuisine = [item.strip() for item in cuisine.split(",") if item.strip()]
    return CoveragePreferences(cuisine=tuple(cuisine))
