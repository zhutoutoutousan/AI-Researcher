import importlib
import inspect
import os
from typing import Dict, Any
# from metachain.util import run_command_in_container
from research_agent.inno.environment.docker_env import DockerEnv
from research_agent.inno.registry import register_tool

@register_tool("check_tool")
def check_tool(env: DockerEnv):
    """
    Extract tools from existing code.
    
    Args:  
    
    Returns:
        A dictionary containing all function definitions {function name: {'source': function source code, 'file': function file path}}
    """
    
    python_script = \
"""import importlib
import inspect
import os
from typing import Dict, Any
def check_tool():
    module = importlib.import_module(f"metachain.tools")
    
    # obtain all function definitions
    functions = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            try:
                # get the source code of the function
                source = inspect.getsource(obj)
                # get the file path of the function definition
                file_path = inspect.getfile(obj)
                functions[name] = {
                    "source": source,
                    "file": file_path
                }
            except Exception as e:
                functions[name] = {
                    "source": f"Failed to get source code: {str(e)}",
                    "file": "Unknown"
                }
            
    return functions
print(check_tool())
"""
    exec_script = f"cd {env.docker_workplace}/metachain && python -c '{python_script.strip()}'"
    response = env.run_command(exec_script)
    if response["status"] == 0:
        return response["result"]
    else:
        return f"Failed to get tool definitions. Error: {response['result']}"

@register_tool("check_agent")
def check_agent(env: DockerEnv):
    """
    Extract agents from existing code.

    Args:  
    
    Returns:
        A dictionary containing all agents definitions {agent name: {'source': agent source code, 'file': agent file path}}
    """
    cmd = f"ls -1 {env.docker_workplace}/metachain/metachain/agents"
    response = env.run_command(cmd)
    if response["status"] == 0:
        agents_files = response["result"].split("\n")
    else: 
        return f"Failed to get agent definitions. Error: {response['result']}"
    agents = {}
    print(agents_files)
    for file in agents_files:
        if file in ["__init__.py", "", "__pycache__"]:
            continue
        cmd = f"cat {env.docker_workplace}/metachain/metachain/agents/{file}"
        response = env.run_command(cmd)
        if response["status"] == 0:
            agent_name = file.split(".")[0]
            agents[agent_name] = {'source': response["result"], 'file': f"{env.docker_workplace}/metachain/metachain/agents/{file}"}
        else:
            return f"Failed to get agent definitions. Error: {response['result']}"
    return agents
