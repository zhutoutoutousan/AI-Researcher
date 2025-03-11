from research_agent.inno.types import Agent
from research_agent.inno.registry import register_agent
from typing import List
def draft_model(idea: str, formulation: str, code: str, reference: List[str]):
    """
    Use this function when you have finished the task and want to give a suggestion about the implementation. You can only use this function after you have checked the implementation and the reference codebases.
    """




@register_agent("get_draft_agent")
def get_draft_agent(model: str, **kwargs):
    def instructions(context_variables):
        pass


