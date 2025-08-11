from research_agent.inno.types import Result, Agent
from typing import Union
from research_agent.inno.registry import register_tool

@register_tool("tool_dummy")
def tool_dummy(args1, args2, context_variables)-> Union[str, Agent, Result]:
    """
    [This is a dummy tool, it's used for demonstrating the usage of the metachain, you should write your own tool instead of using this dummy tool, and the following docstring is just a template, you should modify it to fit your own tool.]
    The tool is used to ...

    Args:
        args1: ...
        args2: ...
        ...
    Returns:
        ...
    Notes:
        The return value can be a string, an agent, or an instance of Result.
    1. The most common return value is a string, but the length of the string should not be too long, and you can save the result to a file if the length is too long.
    2. If the tool is used to transfer the conversation context to an agent, you can return an instance of Agent, like: 
    ```python
    sales_agent = Agent(name="sales_agent", instructions="You are a sales agent.")
    def transfer_to_sales_agent():
        return sales_agent
    ```
    3. If there is some complex operations inside the tool, you can return an instance of Result, for example, you should modify the context variables in the result.
    ```python
    def sales_agent_tool(recommendation: str, context_variables: dict):
        '''
        The tool is used to recommend products to the user.
        '''
        context_variables["recommendation"] = recommendation
        return Result(value="I recommend the product: " + recommendation, agent=user_agent, context_variables=context_variables)
    ```
    4. The tools should be created in the python file in the `metachain/metachain/tools` folder.
    """
    ... # the implementation of the tool
    return "..." # the return value of the tool

