from research_agent.inno.types import Agent
from research_agent.inno.types import Result

from research_agent.inno.tools.terminal_tools import gen_code_tree_structure, read_file, terminal_page_down, terminal_page_up, terminal_page_to
from research_agent.inno.tools.file_surfer_tool import with_env as with_env_file
from research_agent.inno.tools.file_surfer_tool import (
    open_local_file,
    page_up_markdown,
    page_down_markdown,
    find_on_page_ctrl_f,
    find_next,
    visualizer,
    question_answer_on_whole_page
)
from inspect import signature
from research_agent.inno.environment.docker_env import with_env as with_env_docker
from research_agent.inno.environment.docker_env import DockerEnv
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser

def case_resolved(context_variables: dict, analysis_report: str, further_plan: dict[str, str]):
    """
    Use this function to given the analysis report of exsiting experiments and the further plan to the `Machine Learning Agent` to do more experiments. Use this function only after you have carefully and comprehensively reviewed the existing resources and exsiting project as well as fully understand the innovative idea. 

    Args:
        analysis_report (str): The analysis report of exsiting experiments.
        further_plan (dict[str, str]): The further plan to let `Machine Learning Agent` to do more experiments. The key is the name of the experiment, and the value is the description of the experiment.
    """
    if "experiment_report" not in context_variables:
        context_variables["experiment_report"] = []
    context_variables["experiment_report"].append({
        "analysis_report": analysis_report,
        "further_plan": further_plan
    })
    ret_val = f"""\
You have given the analysis report of exsiting experiments and the further plan to the `Machine Learning Agent` to do more experiments.
The analysis report is: {analysis_report}
The further plan is: {further_plan}
"""
    return Result(
        value=ret_val,
        context_variables=context_variables,
    )

def get_exp_analyser_agent(model: str = "gpt-4o", **kwargs):
    file_env: RequestsMarkdownBrowser = kwargs.get("file_env", None)
    assert file_env is not None, "file_env is required"
    code_env: DockerEnv = kwargs.get("code_env", None)
    assert code_env is not None, "code_env is required"
    def instructions(context_variables: dict):
        return """\
You are given an innovative idea and some experimental results conducted by `Machine Learning Agent` in the directory `/workspace/projects/` to implement the idea. You also have some reference codebases and papers in the working directory `/workspace`.
Your task is to: 
1. Analyze the experimental results and give a detailed analysis report about the results.
2. Analyze the reference codebases and papers, and give a further plan to let `Machine Learning Agent` to do more experiments based on the innovative idea. The further experiments could include but not limited to:
    - Modify the implementation to better fit the idea.
    - Add more experiments to prove the effectiveness and superiority of the idea. 
    - Visualize the experimental results and give a detailed analysis report about the results.
    - ANY other experiments that exsiting concurrent reference papers and codebases have done.

AVAILABLE TOOLS:
1. Project and Codebase Navigation:
    - Use `gen_code_tree_structure` to understand codebase structure
    - Use `read_file` to examine specific implementations
    - Use `terminal_page_down`, `terminal_page_up` and `terminal_page_to` to scroll the terminal output when it is too long.
2. Local file navigation:
   - `open_local_file`: Open and read paper files
   - `page_up_markdown`/`page_down_markdown`: Navigate through pages
   - `find_on_page_ctrl_f`/`find_next`: Search specific content
   - `visualizer`: use this tool to SEE the experimental results, the input should be a image or a video and a corresponding question. When the experimental results are image or video, like generated images or the visualization of the experimental results, you should use this tool to see the results and give a detailed analysis report about the results.

[IMPORTANT] You should carefully and comprehensively analyze the experimental results and the reference codebases and papers, and give a detailed analysis report about the results and the further plan by use the `case_resolved` function. DO NOT use this function before you have carefully and comprehensively analyzed the experimental results and the reference codebases and papers.
"""
    tool_files = [
        open_local_file,
        page_up_markdown,
        page_down_markdown,
        find_on_page_ctrl_f,
        find_next,
        question_answer_on_whole_page,
        visualizer,
    ]
    tool_files = [
        with_env_file(file_env)(tool) if "env" in signature(tool).parameters else tool
        for tool in tool_files
    ]

    tool_codes = [
        gen_code_tree_structure,
        read_file,
        terminal_page_down,
        terminal_page_up,
        terminal_page_to
    ]
    tool_codes = [
        with_env_docker(code_env)(tool) if "env" in signature(tool).parameters else tool
        for tool in tool_codes
    ]
    tools = tool_files + tool_codes + [case_resolved]
    return Agent(
        name="Experiment Analysis Agent",
        model=model,
        instructions=instructions,
        functions=tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )
