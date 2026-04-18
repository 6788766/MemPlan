"""
WorkBench toolkits bridge for MemPlan.

WorkBench's inference loop expects LangChain tools; MemPlan's WorkBench tools are
wrapped in a minimal Tool object (`task_helper.work.tools.tooling.Tool`).

We convert those Tool objects into LangChain `StructuredTool` objects for the agent,
while keeping the original module-level Tool objects for evaluation/replay.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from . import (
    analytics,
    calendar,
    company_directory,
    customer_relationship_manager,
    email,
    project_management,
)


def _as_structured(tool_obj) -> StructuredTool:
    name = getattr(tool_obj, "name", None)
    func = getattr(tool_obj, "func", None)
    return_direct = bool(getattr(tool_obj, "return_direct", False))
    if not isinstance(name, str) or not callable(func):
        raise TypeError(f"Expected a Tool-like object with .name/.func, got: {tool_obj!r}")
    desc = getattr(func, "__doc__", None) or name
    return StructuredTool.from_function(func=func, name=name, description=str(desc), return_direct=return_direct)


# Lists of Tool objects (for metrics like side effects).
tools_with_side_effects = [
    calendar.create_event,
    calendar.delete_event,
    calendar.update_event,
    email.send_email,
    email.delete_email,
    email.forward_email,
    email.reply_email,
    analytics.create_plot,
    project_management.create_task,
    project_management.delete_task,
    project_management.update_task,
    customer_relationship_manager.update_customer,
    customer_relationship_manager.add_customer,
    customer_relationship_manager.delete_customer,
]

tools_without_side_effects = [
    calendar.get_event_information_by_id,
    calendar.search_events,
    email.get_email_information_by_id,
    email.search_emails,
    analytics.engaged_users_count,
    analytics.get_visitor_information_by_id,
    analytics.traffic_source_count,
    analytics.total_visits_count,
    analytics.get_average_session_duration,
    project_management.get_task_information_by_id,
    project_management.search_tasks,
    customer_relationship_manager.search_customers,
    company_directory.find_email_address,
]

all_tools = tools_with_side_effects + tools_without_side_effects

tool_information = [
    {
        "toolkit": getattr(tool, "__module__", ""),
        "tool": tool,
        "name": getattr(tool, "name", ""),
    }
    for tool in all_tools
]

# LangChain tools for inference.
_all_structured = [_as_structured(t["tool"]) for t in tool_information]

calendar_toolkit = [t for t in _all_structured if t.name.split(".")[0] == "calendar"]
email_toolkit = [t for t in _all_structured if t.name.split(".")[0] == "email"]
analytics_toolkit = [t for t in _all_structured if t.name.split(".")[0] == "analytics"]
project_management_toolkit = [t for t in _all_structured if t.name.split(".")[0] == "project_management"]
customer_relationship_manager_toolkit = [
    t for t in _all_structured if t.name.split(".")[0] == "customer_relationship_manager"
]
company_directory_toolkit = [t for t in _all_structured if t.name.split(".")[0] == "company_directory"]
