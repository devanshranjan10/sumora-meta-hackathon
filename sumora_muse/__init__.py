"""Sumora's Meta Model API layer: Muse Spark routing, agents with approval gates, Muse Image and Muse Voice."""

from .catalog import Catalogue, read_menu_photo
from .inspect import Inspection, inspect_fabric, merge_boxes
from .agent import AgentRun, ApprovalRequest, Tool, run_agent
from .client import ChatResult, MuseClient, MuseError
from .router import Router, RunRecord
from .team import Team, plan_team

__all__ = [
    "AgentRun", "Catalogue", "read_menu_photo", "Inspection", "inspect_fabric", "merge_boxes", "ApprovalRequest", "ChatResult", "MuseClient", "MuseError",
    "Router", "RunRecord", "Team", "Tool", "plan_team", "run_agent",
]
__version__ = "0.1.0"
