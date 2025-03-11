from research_agent.inno.registry import register_tool
from browsergym.core.action.highlevel import HighLevelActionSet
from typing import Literal
from research_agent.inno.environment.browser_env import BrowserEnv, VIEWPORT
from research_agent.inno.environment.docker_env import DockerEnv, DockerConfig
from browsergym.utils.obs import flatten_axtree_to_str
from dataclasses import dataclass, field
from typing import Dict
from urllib.parse import quote_plus
from research_agent.inno.types import Result
from functools import partial, update_wrapper
from inspect import signature
import tiktoken
from datetime import datetime
from research_agent.inno.util import function_to_json
# def with_env(env: BrowserEnv):
#     """将env注入到工具函数中的装饰器"""
#     def decorator(func):
#         # 创建新函数，固定env参数
#         new_func = partial(func, env=env)
#         # 保留原始函数的docstring和signature
#         update_wrapper(new_func, func)
#         # 修改signature，移除env参数
#         new_func.__signature__ = signature(func).replace(
#             parameters=[p for p in signature(func).parameters.values() if p.name != 'env']
#         )
#         return new_func
#     return decorator
def with_env(env: BrowserEnv):
    """将env注入到工具函数中的装饰器"""
    def decorator(func):
        def wrapped(*args, **kwargs):
            return func(env=env, *args, **kwargs)
        
        # 保留原始函数的所有属性
        update_wrapper(wrapped, func)
        # 修改signature，移除env参数
        wrapped.__signature__ = signature(func).replace(
            parameters=[p for p in signature(func).parameters.values() if p.name != 'env']
        )
        return wrapped
    return decorator

def with_two_envs(env: BrowserEnv, code_env: DockerEnv):
    """将env注入到工具函数中的装饰器"""
    def decorator(func):
        def wrapped(*args, **kwargs):
            return func(env=env, code_env=code_env, *args, **kwargs)
        
        # 保留原始函数的所有属性
        update_wrapper(wrapped, func)
        # 修改signature，移除env参数
        wrapped.__signature__ = signature(func).replace(
            parameters=[p for p in signature(func).parameters.values() if p.name not in ['env', 'code_env']]
        )
        return wrapped
    return decorator
@dataclass
class WebObservation:
    content: str  # text content of the page
    url: str # URL of the page
    screenshot: str  # base64-encoded screenshot, png
    open_pages_urls: list[str] # list of open pages
    active_page_index: int  # index of the active page
    dom_object: dict  # DOM object
    axtree_object: dict  # accessibility tree object
    extra_element_properties: dict
    focused_element_bid: str  # focused element bid
    last_browser_action: str  # last browser env action performed
    last_browser_action_error: str # last browser env action error
    error: bool  # error flag

def to_web_obs(obs) -> WebObservation:
    obs_dict = dict(
        content=obs['text_content'],  # text content of the page
        url=obs.get('url', ''),  # URL of the page
        # screenshot=obs.get('screenshot', None),  # base64-encoded screenshot, png
        screenshot=None,  # base64-encoded screenshot, png
        open_pages_urls=obs.get('open_pages_urls', []),  # list of open pages
        active_page_index=obs.get(
            'active_page_index', -1
        ),  # index of the active page
        dom_object=obs.get('dom_object', {}),  # DOM object
        axtree_object=obs.get('axtree_object', {}),  # accessibility tree object
        extra_element_properties=obs.get('extra_element_properties', {}),
        focused_element_bid=obs.get(
            'focused_element_bid', None
        ),  # focused element bid
        last_browser_action=obs.get(
            'last_action', ''
        ),  # last browser env action performed
        last_browser_action_error=obs.get('last_action_error', ''),
        error=True if obs.get('last_action_error', '') else False,  # error flag
    )
    return WebObservation(**obs_dict)
def wrap_return_value(web_obs: WebObservation, action_description: str = ""):
    error_prefix = ""
    if web_obs.error:
        error_prefix = get_error_prefix(web_obs.last_browser_action, web_obs.last_browser_action_error)
    cur_url = web_obs.url
    try:
        cur_axtree_txt = flatten_axtree_to_str(
            web_obs.axtree_object,
            extra_properties=web_obs.extra_element_properties,
            with_clickable=True,
            filter_visible_only=True,
        )
    except Exception as e:
        cur_axtree_txt = f'Error encountered when browsing.\nError when trying to process the accessibility tree:{str(e)}'
    ret_value = f"""\
{error_prefix}
{action_description}

# Current Page URL:
{cur_url}

# Current Accessibility Tree:
{cur_axtree_txt}

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12
```click("12")```
"
""".strip()
    return ret_value
def get_error_prefix(last_browser_action: str, last_browser_action_error: str) -> str:
    return f'IMPORTANT! Last action is incorrect:\n{last_browser_action}\nThink again with the current observation of the page.\nThe error message is:\n{last_browser_action_error}'

# @register_tool("click")
# def click(env: BrowserEnv, bid: str, button: Literal["left", "middle", "right"] = "left", modifiers: list[Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]] = []):
#     """
#     Clicks the mouse on the target with the given element bid.
#     Args:
#         bid: The bid of the element to click.
#         button: The button to click.
#         modifiers: The modifiers to click.
#     """
#     try:
#         # 执行动作
#         # action = action_func(*args, **kwargs)
#         button_str = f''', button="{button}"''' if button else ''
#         modifiers_str = f', modifiers={modifiers}' if modifiers else ''
#         action_str = f"""click('{bid}'{button_str}{modifiers_str})"""
        
#         # 与环境交互
#         obs = env.step(action_str)
#         web_obs = to_web_obs(obs)
        
#     except Exception as e:
#         return f"Error encountered when taking action: {action_str}\nError: {e}"
#     ret_value = wrap_return_value(web_obs)
#     return Result(
#             value=ret_value,
#             image=web_obs.screenshot, 
#         )
@register_tool("click")
def click(env: BrowserEnv, bid: str, button: Literal["left", "middle", "right"] = "left"):
    """
    Clicks the mouse on the target with the given element bid.
    Args:
        bid: The bid of the element to click.
        button: The button to click.
    """
    try:
        # 执行动作
        # action = action_func(*args, **kwargs)
        button_str = f''', button="{button}"''' if button else ''
        action_str = f"""_click_id('{bid}'{button_str})"""
        
        # 与环境交互
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
        
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("page_down")
def page_down(env: BrowserEnv):
    """
    Scrolls the entire browser viewport one page DOWN towards the end.
    """
    try:    
        action_str = f'scroll(0, {VIEWPORT["height"]-50})'
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("page_up")
def page_up(env: BrowserEnv):
    """
    Scrolls the entire browser viewport one page UP towards the beginning.
    """
    try:    
        action_str = f'scroll(0, -{VIEWPORT["height"]-50})'
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("history_back")
def history_back(env: BrowserEnv):
    """
    Navigates back one page in the browser's history. This is equivalent to clicking the browser back button.
    """
    try:
        action_str = 'go_back()'
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("history_forward")
def history_forward(env: BrowserEnv):
    """
    Navigates forward one page in the browser's history. This is equivalent to clicking the browser forward button.
    """
    try:
        action_str = 'go_forward()'
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("input_text")
def input_text(env: BrowserEnv, bid: str, text: str):
    """
    Types the given text value into the specified field.
    Args:
        bid: The bid of the element to type into.
        text: The text to type into the input field.
    """
    try:
        action_str = f"fill('{bid}', '{text}')"
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )

@register_tool("visit_url")
def visit_url(env: BrowserEnv, url: str): 
    """
    Navigate directly to a provided URL using the browser's address bar. Prefer this tool over other navigation techniques in cases where the user provides a fully-qualified URL (e.g., choose it over clicking links, or inputing queries into search boxes).
    Args:
        url: The URL to navigate to.
    """
    try:
        if url.startswith(("https://", "http://", "file://", "about:")):
            action_str = f"_visit_page('{url}')"
            obs = env.step(action_str)
            web_obs = to_web_obs(obs)
        elif " " in url:
            query = quote_plus(url)
            action_str = f"_visit_page('https://www.google.com.sg/search?q={query}&hl=en&gl=US')"
            obs = env.step(action_str)
            web_obs = to_web_obs(obs)
        else:
            action_str = f"_visit_page('https://{url}')"
            obs = env.step(action_str)
            web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )

@register_tool("web_search")
def web_search(env: BrowserEnv, query: str):
    """
    Performs a web search on 'https://www.google.com.sg/?hl=en&gl=US' with the given query.
    Args:
        query: The query to search for.
    """
    try:
        action_str = f"_visit_page('https://www.google.com.sg/search?q={quote_plus(query)}&hl=en&gl=US')"
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("sleep")
def sleep(env: BrowserEnv):
    """
    Wait a short period of time. Call this function if the page has not yet fully loaded, or if it is determined that a small delay would increase the task's chances of success.
    """
    try: 
        action_str = f"noop(3000)"
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
def truncate_by_tokens(env: DockerEnv, text, max_tokens = 4096, model="gpt-4o-2024-08-06"):
    from inno.tools.files import create_file, create_directory
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_directory(f"{env.docker_workplace}/web_page_markdown_output", env=env)
    output_path = f"{env.docker_workplace}/web_page_markdown_output/full_output_{timestamp}.md"
    create_msg = create_file(output_path, content = text, env=env)
    # 截断tokens并解码回字符串
    truncated_tokens_bos = tokens[:max_tokens//2]
    truncated_tokens_eos = tokens[-(max_tokens - len(truncated_tokens_bos)):]
    if create_msg.startswith("Error"):
        return encoding.decode(truncated_tokens_bos) + "\n...\n" + encoding.decode(truncated_tokens_eos) + "\n\nThe full convert markdown output is too long, so I want to save them into the file: {0}\n\nBut I got an error: {1}".format(output_path, create_msg)
    
    return encoding.decode(truncated_tokens_bos) + "\n...\n" + encoding.decode(truncated_tokens_eos) + "\n\nThe full convert markdown output is too long, so it is saved in the file: {0}\n\nYou may use the `File Surfer Agent` to view the full output.".format(output_path)

@register_tool("get_page_markdown")
def get_page_markdown(env: BrowserEnv, code_env: DockerEnv):
    """
    Get the markdown content of the current page. 
    Use this tool if you need to watch the Youtube video, Wikipedia page, or other pages that contain media content. 
    Note that this tool can only be used after you have visited a valid page.
    """
    try:
        action_str = "_get_page_markdown()"
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
        obs = env.step("go_back()")
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"

    ret_value = \
f"""
I have converted the current page into clean markdown format:
{web_obs.content}
"""
    ret_value = truncate_by_tokens(code_env, ret_value, max_tokens=10000)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )

if __name__ == "__main__":
    env = BrowserEnv(browsergym_eval_env = None, local_root="/Users/tangjiabin/Documents/reasoning/metachain", workplace_name="workplace_gaia_eval")
    # code_env = DockerEnv(DockerConfig(container_name = "gaia_lite_eval", 
    # workplace_name = "workplace_gaia_eval", 
    # communication_port = 12345, 
    # conda_path = "/root/miniconda3"))
    # code_env.init_container()
    # import json
    # web_search_with_env = with_env(env)(web_search)
    # print(json.dumps(function_to_json(web_search_with_env), indent=4))
    # visit_url(env, "https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=LLMRec&oq=")
    # res = page_down(env)
    # print(res.value)
    # res = visit_url(env, 'https://arxiv.org/pdf/2310.13023')
    # print(res.value)

    res = visit_url(env, 'https://www.collinsdictionary.com/dictionary/italian-english/cumulo')
    # res = visit_url(env, 'https://www.reddit.com/r/ChatGPT/comments/1h5ey4m/chatgpt_helped_me_not_blow_up_on_my_boss/')
    print(res.value)  
    # bid = input("Please input the bid of the element to click: ")
    # res = click(env, bid)
    # print(res.value)
    # res = get_page_markdown(env, code_env)
    # print(res.value)
    # res = page_down(env)
    # print(res.value)


"""
    Example: If you are required to watch a Youtube video, you should first visit the page containing the video using `visit_url` tool, and then you can use this tool to convert the page content to markdown format, and then you extract information from the video description, transcript and son on, which is equal to watching the video. Other pages that contain media content are similar.
"""