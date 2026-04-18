"""
Bridge package for WorkBench tool imports.

Allows: `from baseline.workbench.src.tools import calendar, email, ...`
"""

from . import (  # noqa: F401
    analytics,
    calendar,
    company_directory,
    customer_relationship_manager,
    email,
    project_management,
)
