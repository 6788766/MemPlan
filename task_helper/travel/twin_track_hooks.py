from __future__ import annotations

import math
import re
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


def _is_placeholder(candidate: Mapping[str, object]) -> bool:
    return str(candidate.get("source") or "") == "placeholder" or str(candidate.get("text") or "") == "-"


def _safe_float(value: object, default: Optional[float] = 0.0) -> Optional[float]:
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return float(value)
        number = float(value)  # type: ignore[arg-type]
        if math.isnan(number) or math.isinf(number):
            return default
        return float(number)
    except (TypeError, ValueError):
        return default


def _travel_parse_day_from_phase(phase_key: str, phase_index: int) -> int:
    if phase_index:
        return int(phase_index)
    match = re.search(r"(\d+)$", phase_key)
    if match:
        return int(match.group(1))
    return 0


def _travel_canon_city_name(value: str) -> str:
    if not value:
        return value
    idx = value.find("(")
    if idx == -1:
        return value.strip()
    return value[:idx].strip()


def _travel_query_has_city_placeholder(slot) -> bool:
    params = slot.query_action.get("params") if isinstance(slot.query_action.get("params"), dict) else {}
    for key in ("current_city", "origin", "destination", "City", "city"):
        raw = params.get(key)
        if isinstance(raw, str) and "CITY_" in raw:
            return True
    return False


def _travel_extract_restaurant_key(slot, candidate: Mapping[str, object]) -> Optional[str]:
    if slot.action_type != "Eat" or _is_placeholder(candidate):
        return None
    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    if isinstance(meta, dict) and meta.get("type") == "memory":
        attrs = meta.get("attrs") if isinstance(meta.get("attrs"), dict) else {}
        params = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
        name = params.get("Name")
        city = params.get("City")
        if name and city:
            return f"{name}, {city}"
    row = meta.get("row") if isinstance(meta.get("row"), dict) else {}
    name = row.get("Name")
    city = row.get("City")
    if name and city:
        return f"{name}, {city}"
    return None


def _travel_extract_attraction_key(slot, candidate: Mapping[str, object]) -> Optional[str]:
    if slot.action_type != "Visit" or _is_placeholder(candidate):
        return None
    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    if isinstance(meta, dict) and meta.get("type") == "memory":
        attrs = meta.get("attrs") if isinstance(meta.get("attrs"), dict) else {}
        params = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
        name = params.get("Name")
        city = params.get("City")
        if name and city:
            return f"{name}, {city}"
    row = meta.get("row") if isinstance(meta.get("row"), dict) else {}
    name = row.get("Name")
    city = row.get("City")
    if name and city:
        return f"{name}, {city}"
    return None


def _travel_extract_transport_mode(slot, candidate: Mapping[str, object]) -> Optional[str]:
    if slot.action_type != "Move" or _is_placeholder(candidate):
        return None
    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    if isinstance(meta, dict) and meta.get("type") == "memory":
        attrs = meta.get("attrs") if isinstance(meta.get("attrs"), dict) else {}
        mode = attrs.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
    if isinstance(meta, dict) and meta.get("type") == "distance":
        mode = meta.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
    if isinstance(meta, dict) and meta.get("type") == "flight":
        return "Flight"
    return None


def _travel_transport_restriction(plan: Mapping[str, object]) -> Optional[str]:
    local = plan.get("local_constraint")
    if isinstance(local, dict):
        value = local.get("transportation")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def online_step(state, *, slot, candidate: Mapping[str, object]) -> bool:
    """
    TravelPlanner online pruning and per-rollout bookkeeping.

    `state` is the planner's MonitorState (task-agnostic) with a mutable `task_state` dict.
    """

    task_state = getattr(state, "task_state", None)
    if not isinstance(task_state, dict):
        return False

    used_restaurants: Set[str] = task_state.setdefault("used_restaurants", set())
    used_attractions: Set[str] = task_state.setdefault("used_attractions", set())
    transport_modes: Set[str] = task_state.setdefault("transport_modes", set())
    city_assignments: Dict[str, str] = task_state.setdefault("city_assignments", {})
    used_assigned_cities: Set[str] = task_state.setdefault("used_assigned_cities", set())

    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    assignments = meta.get("assignments") if isinstance(meta, dict) else None

    # Placeholder cities (CITY_k(State)) must be grounded consistently across the plan.
    if _travel_query_has_city_placeholder(slot):
        if not isinstance(assignments, dict) or not assignments:
            return False
        for token, city in assignments.items():
            if not isinstance(token, str) or not token:
                continue
            if not isinstance(city, str) or not city:
                continue
            canon = _travel_canon_city_name(city)
            existing = city_assignments.get(token)
            if existing is not None:
                if _travel_canon_city_name(existing) != canon:
                    return False
                continue
            if canon in used_assigned_cities:
                return False
            city_assignments[token] = city
            used_assigned_cities.add(canon)

    # DayRecord should preserve the move-day marker ("from ... to ...") vs non-move day.
    if slot.action_type == "DayRecord":
        params = slot.query_action.get("params") if isinstance(slot.query_action.get("params"), dict) else {}
        raw_current = params.get("current_city")
        if isinstance(raw_current, str) and raw_current.strip():
            expected_move = "from " in raw_current.lower()
            cand_current: Optional[str] = None
            if isinstance(meta, dict) and meta.get("type") == "city_search":
                cand_current = meta.get("current_city") if isinstance(meta.get("current_city"), str) else None
            elif isinstance(meta, dict) and meta.get("type") == "memory":
                attrs = meta.get("attrs") if isinstance(meta.get("attrs"), dict) else {}
                mem_params = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
                cand_current = mem_params.get("current_city") if isinstance(mem_params.get("current_city"), str) else None
            if isinstance(cand_current, str) and cand_current.strip():
                cand_move = "from " in cand_current.lower()
                if expected_move != cand_move:
                    return False

    # Diversity constraints (cheap pruning).
    rest_key = _travel_extract_restaurant_key(slot, candidate)
    if rest_key:
        if rest_key in used_restaurants:
            return False
        used_restaurants.add(rest_key)

    attr_key = _travel_extract_attraction_key(slot, candidate)
    if attr_key:
        if attr_key in used_attractions:
            return False
        used_attractions.add(attr_key)

    # Transportation constraints.
    mode = _travel_extract_transport_mode(slot, candidate)
    if mode:
        if mode == "Self-driving" and ("Flight" in transport_modes or "Taxi" in transport_modes):
            return False
        if mode == "Flight" and "Self-driving" in transport_modes:
            return False
        if mode == "Taxi" and "Self-driving" in transport_modes:
            return False

        restriction = _travel_transport_restriction(state.plan)
        if restriction == "no flight" and mode == "Flight":
            return False
        if restriction == "no self-driving" and mode == "Self-driving":
            return False
        transport_modes.add(mode)

    return True


def fill_action(
    query_action: Mapping[str, object],
    candidate: Mapping[str, object],
    slot,
) -> Dict[str, object]:
    filled: Dict[str, object] = dict(query_action)
    base_params = query_action.get("params") if isinstance(query_action.get("params"), dict) else {}
    params: Dict[str, object] = dict(base_params)

    def _merge_params(update: Mapping[str, object]) -> None:
        for key, value in update.items():
            if value is None:
                continue
            params[key] = value

    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    meta_type = meta.get("type") if isinstance(meta, dict) else None

    if _is_placeholder(candidate):
        filled["params"] = params
        return filled

    if meta_type == "memory":
        attrs = meta.get("attrs") if isinstance(meta.get("attrs"), dict) else {}
        mem_params = attrs.get("params") if isinstance(attrs.get("params"), dict) else {}
        _merge_params(mem_params)
        for key, value in attrs.items():
            if key in {"action_type", "params", "order_index"}:
                continue
            if value is not None:
                filled[key] = value
        filled["params"] = params
        return filled

    # Tool candidates.
    if slot.action_type == "DayRecord" and meta_type == "city_search":
        current = meta.get("current_city")
        if isinstance(current, str) and current.strip():
            params["current_city"] = current.strip()
        filled["params"] = params
        return filled

    if slot.action_type == "Move":
        if meta_type == "flight":
            row = meta.get("row") if isinstance(meta.get("row"), dict) else {}
            origin = meta.get("origin") or params.get("origin")
            destination = meta.get("destination") or params.get("destination")
            flight_no = row.get("Flight Number") or row.get("FlightNumber") or row.get("flight_no")
            dep = row.get("DepTime")
            arr = row.get("ArrTime")
            _merge_params(
                {
                    "origin": origin,
                    "destination": destination,
                    "Flight Number": flight_no,
                    "DepTime": dep,
                    "ArrTime": arr,
                    "ActualElapsedTime": row.get("ActualElapsedTime"),
                    "FlightDate": row.get("FlightDate") or meta.get("date"),
                    "distance": row.get("Distance"),
                    "cost": row.get("Price"),
                }
            )
            filled["mode"] = "Flight"
            filled["raw_transportation"] = (
                f"Flight Number: {flight_no}, from {origin} to {destination}, "
                f"Departure Time: {dep}, Arrival Time: {arr}"
            )
        elif meta_type == "distance":
            origin = meta.get("origin") or params.get("origin")
            destination = meta.get("destination") or params.get("destination")
            info = meta.get("info") if isinstance(meta.get("info"), dict) else {}
            mode = meta.get("mode")
            _merge_params(
                {
                    "origin": origin,
                    "destination": destination,
                    "duration": info.get("duration"),
                    "distance": info.get("distance"),
                    "cost": info.get("cost"),
                }
            )
            if isinstance(mode, str) and mode:
                filled["mode"] = mode
            filled["raw_transportation"] = (
                f"{filled.get('mode') or 'Self-driving'}, from {origin} to {destination}, "
                f"duration: {info.get('duration')}, distance: {info.get('distance')}, cost: {info.get('cost')}"
            )
        filled["params"] = params
        return filled

    if slot.action_type in {"Eat", "Visit", "Stay"}:
        row = meta.get("row") if isinstance(meta.get("row"), dict) else {}
        _merge_params(row)
        filled["params"] = params
        return filled

    filled["params"] = params
    return filled


def build_eval_plan(
    plan: Mapping[str, object],
    phases,
    slots,
    chosen_actions: Sequence[Dict[str, object]],
) -> List[dict]:
    days = int(_safe_float(plan.get("days"), default=0) or 0)
    days = max(days, 0)
    records: List[dict] = []
    attractions_by_day: Dict[int, List[str]] = {d: [] for d in range(1, days + 1)}
    for d in range(1, days + 1):
        records.append(
            {
                "days": d,
                "current_city": "-",
                "transportation": "-",
                "breakfast": "-",
                "lunch": "-",
                "dinner": "-",
                "attraction": "-",
                "accommodation": "-",
            }
        )

    for slot, action in zip(slots, chosen_actions):
        day = _travel_parse_day_from_phase(slot.phase_key, slot.phase_index)
        if not day or day < 1 or day > days:
            continue
        record = records[day - 1]
        action_type = str(action.get("action_type") or "")
        params = action.get("params") if isinstance(action.get("params"), dict) else {}

        if action_type == "DayRecord":
            current = params.get("current_city")
            if isinstance(current, str) and current.strip():
                record["current_city"] = current.strip()
            continue

        if action_type == "Move":
            raw = action.get("raw_transportation")
            if isinstance(raw, str) and raw.strip():
                record["transportation"] = raw.strip()
                continue
            mode = action.get("mode") or ""
            origin = params.get("origin")
            destination = params.get("destination")
            if isinstance(mode, str) and mode.lower() == "flight":
                flight_no = params.get("Flight Number")
                dep = params.get("DepTime")
                arr = params.get("ArrTime")
                record["transportation"] = (
                    f"Flight Number: {flight_no}, from {origin} to {destination}, "
                    f"Departure Time: {dep}, Arrival Time: {arr}"
                )
            else:
                record["transportation"] = (
                    f"{mode or 'Self-driving'}, from {origin} to {destination}, "
                    f"duration: {params.get('duration')}, distance: {params.get('distance')}, cost: {params.get('cost')}"
                )
            continue

        if action_type == "Eat":
            name = params.get("Name")
            city = params.get("City") or params.get("city")
            slot_name = action.get("slot")
            if not slot_name:
                attrs = action.get("attrs")
                if isinstance(attrs, dict):
                    slot_name = attrs.get("slot")
            if name and city:
                key = str(slot_name or "").lower()
                if key == "breakfast":
                    record["breakfast"] = f"{name}, {city}"
                elif key == "lunch":
                    record["lunch"] = f"{name}, {city}"
                elif key == "dinner":
                    record["dinner"] = f"{name}, {city}"
            continue

        if action_type == "Visit":
            name = params.get("Name")
            city = params.get("City") or params.get("city")
            if name and city:
                attractions_by_day.setdefault(day, []).append(f"{name}, {city}")
            continue

        if action_type == "Stay":
            name = params.get("NAME")
            city = params.get("city") or params.get("City")
            if name and city:
                record["accommodation"] = f"{name}, {city}"
            continue

    for day in range(1, days + 1):
        items = attractions_by_day.get(day, [])
        if items:
            records[day - 1]["attraction"] = ";".join(items) + ";"
    return records


def _travel_build_query_data(plan: Mapping[str, object]) -> Dict[str, object]:
    query = dict(plan)
    # The official TravelPlanner constraint checkers expect certain keys to exist
    # and use direct indexing (e.g., `question['org']`). Be defensive here so
    # scoring (MCTS + LLM repair) never crashes on missing keys.
    if "org" not in query:
        query["org"] = query.get("origin") or query.get("from") or "-"
    if "dest" not in query:
        query["dest"] = query.get("destination") or query.get("to") or "-"
    if "visiting_city_number" not in query:
        query["visiting_city_number"] = query.get("visiting_city_num") or 1
    try:
        query["days"] = int(query.get("days") or 0)
    except (TypeError, ValueError):
        query["days"] = 0
    try:
        query["people_number"] = int(query.get("people_number") or 1)
    except (TypeError, ValueError):
        query["people_number"] = 1
    try:
        query["visiting_city_number"] = int(query.get("visiting_city_number") or 1)
    except (TypeError, ValueError):
        query["visiting_city_number"] = 1
    if int(query["visiting_city_number"]) <= 0:
        query["visiting_city_number"] = 1
    try:
        query["budget"] = float(query.get("budget") or 0.0)
    except (TypeError, ValueError):
        query["budget"] = 0.0
    local = query.get("local_constraint")
    if not isinstance(local, dict):
        local = {}
    # Required keys for the official evaluator.
    for key in ("house rule", "cuisine", "room type", "transportation"):
        local.setdefault(key, None)
    query["local_constraint"] = local
    return query


def evaluate(
    plan: Mapping[str, object],
    phases,
    slots,
    chosen_indices: Sequence[int],
    config,
):
    from planner.twin_track import ScoreResult
    from task_helper.travel.evaluation import commonsense_constraint, hard_constraint  # local import: heavy CSV load

    chosen_actions = [
        fill_action(slot.query_action, slot.candidates[idx], slot) if idx < len(slot.candidates) else dict(slot.query_action)
        for slot, idx in zip(slots, chosen_indices)
    ]
    eval_plan = build_eval_plan(plan, phases, slots, chosen_actions)
    query_data = _travel_build_query_data(plan)

    semantic_info = commonsense_constraint.evaluation(query_data, eval_plan)
    flags = [entry[0] for entry in semantic_info.values() if entry[0] is not None]
    semantic_rate = sum(1 for flag in flags if flag) / len(flags) if flags else 1.0

    delivery_ok = bool(
        isinstance(semantic_info, dict)
        and semantic_info.get("is_not_absent", (False, None))[0]
        and semantic_info.get("is_valid_information_in_sandbox", (False, None))[0]
    )

    hard_info: Dict[str, Tuple[Optional[bool], Optional[str]]] = {}
    hard_pass = False
    if delivery_ok:
        hard_info_raw = hard_constraint.evaluation(query_data, eval_plan)
        if isinstance(hard_info_raw, dict):
            hard_info = {k: (v[0], v[1]) for k, v in hard_info_raw.items()}
            hard_pass = all(entry[0] for entry in hard_info.values() if entry[0] is not None)

    total_cost = float(hard_constraint.get_total_cost(query_data, eval_plan) or 0.0)

    tolerance = float(getattr(config, "semantic_tolerance", 0.8))
    tolerance = max(0.0, min(1.0, tolerance))
    hard_fail_penalty = float(getattr(config, "hard_fail_penalty", -1e12))
    shortfall_weight = float(getattr(config, "semantic_shortfall_weight", 1e6))
    if not hard_pass:
        reward = hard_fail_penalty - total_cost
    elif semantic_rate < tolerance:
        reward = (-total_cost) - (shortfall_weight * (tolerance - semantic_rate))
    else:
        reward = -total_cost

    semantic_details = {k: (v[0], v[1]) for k, v in semantic_info.items()} if isinstance(semantic_info, dict) else {}
    return ScoreResult(
        reward=float(reward),
        hard_pass=bool(hard_pass),
        semantic_rate=float(semantic_rate),
        total_cost=float(total_cost),
        hard_details=hard_info,
        semantic_details=semantic_details,
    )
