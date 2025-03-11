import click
import importlib
from research_agent.inno import MetaChain
from research_agent.inno.util import debug_print
@click.group()
def cli():
    """The command line interface for metachain"""
    pass

@cli.command()
@click.option('--model', default='gpt-4o-2024-08-06', help='the name of the model')
@click.option('--agent_func', default='get_dummy_agent', help='the function to get the agent')
@click.option('--query', default='...', help='the user query to the agent')
@click.argument('context_variables', nargs=-1)
def agent(model: str, agent_func: str, query: str, context_variables):
    """
    Run an agent with a given model, agent function, query, and context variables.
    Args:
        model (str): The name of the model.
        agent_func (str): The function to get the agent.
        query (str): The user query to the agent.
        context_variables (list): The context variables to pass to the agent.
    Usage:
        mc agent --model=gpt-4o-2024-08-06 --agent_func=get_weather_agent --query="What is the weather in Tokyo?" city=Tokyo unit=C timestamp=2024-01-01
    """ 
    context_storage = {}
    for arg in context_variables:
        if '=' in arg:
            key, value = arg.split('=', 1)
            context_storage[key] = value
    agent_module = importlib.import_module(f'metachain.agents')
    try:
        agent_func = getattr(agent_module, agent_func)
    except AttributeError:
        raise ValueError(f'Agent function {agent_func} not found, you shoud check in the `metachain.agents` directory for the correct function name')
    agent = agent_func(model)
    mc = MetaChain()
    messages = [
        {"role": "user", "content": query}
    ]
    response = mc.run(agent, messages, context_storage, debug=True)
    debug_print(True, response.messages[-1]['content'], title = f'Result of running {agent.name} agent', color = 'pink3')
    return response.messages[-1]['content']

