from research_agent.inno.registry import register_tool

@register_tool("case_resolved")
def case_resolved(result: str):
    """
    Use this function when the case is resolved and no further actions are needed. Please encapsulate your final answer (answer ONLY) within <solution> and </solution>. 

    Args:
        result: The final result of the case resolution following the instructions.

    Example: case_resolved(`The answer to the question is <solution> 42 </solution>`)
    """
    return f"Case resolved. No further actions are needed. The result of the case resolution is: {result}"

@register_tool("case_not_resolved")
def case_not_resolved(failure_reason: str):
    """
    Use this function when the case is not resolved when all agents have tried their best.
    [IMPORTANT] Please do not use this function unless all of you have tried your best.

    Args:
        failure_reason: The reason why the case is not resolved.
    """
    return f"Case not resolved. No further actions are needed. The reason is: {failure_reason}"

if __name__ == "__main__":
    print(case_resolved())