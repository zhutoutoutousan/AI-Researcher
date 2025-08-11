from research_agent.inno.registry import register_tool
from browsergym.core.action.highlevel import HighLevelActionSet
from typing import Literal
from research_agent.inno.environment.browser_env import BrowserEnv, VIEWPORT
from research_agent.inno.environment.docker_env import DockerEnv, DockerConfig
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str
from dataclasses import dataclass, field
from typing import Dict, List
from urllib.parse import quote_plus
from research_agent.inno.types import Result
from functools import partial, update_wrapper
from inspect import signature
import tiktoken
from datetime import datetime
from collections import defaultdict
from research_agent.inno.util import function_to_json
from research_agent.inno.environment.browser_cookies import get_all_cookies
import requests
import re
import os
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
def flatten_google_scholar_results(dom_snapshot, extra_properties: dict = None) -> list:
    """extract the paper information from the Google Scholar search results, return the structured list of dictionaries"""
    if not dom_snapshot or "documents" not in dom_snapshot or not dom_snapshot["documents"]:
        return []
    
    def to_string(idx):
        if idx == -1:
            return None
        else:
            return dom_snapshot["strings"][idx]

    def parse_document(document_idx) -> list:
        if "nodes" not in dom_snapshot["documents"][document_idx]:
            return []
        nodes = dom_snapshot["documents"][document_idx]["nodes"]
        node_children = defaultdict(list)
        results = []

        # 构建节点树
        for node_idx in range(len(nodes["nodeName"])):
            parent_idx = nodes["parentIndex"][node_idx]
            if parent_idx != -1:
                node_children[parent_idx].append(node_idx)

        def get_node_text(node_idx) -> str:
            text = ""
            node_type = nodes["nodeType"][node_idx]
            node_value = to_string(nodes["nodeValue"][node_idx])
            
            if node_type == 3 and node_value:  # 文本节点
                text += node_value.strip() + " "
            
            for child_idx in node_children[node_idx]:
                text += get_node_text(child_idx)
            return text.strip()

        def get_node_links(node_idx) -> list:
            """获取节点内所有的链接"""
            links = []
            if nodes["nodeType"][node_idx] == 1:  # 元素节点
                # 检查当前节点是否为链接
                attrs = nodes["attributes"][node_idx]
                href = None
                text = get_node_text(node_idx)
                
                for i in range(0, len(attrs), 2):
                    if to_string(attrs[i]) == "href":
                        href = to_string(attrs[i + 1])
                        if href:
                            links.append({"text": text, "href": href})
                            break
                
                # 递归检查子节点
                for child_idx in node_children[node_idx]:
                    links.extend(get_node_links(child_idx))
            return links

        def find_node_by_class(root_idx, target_class):
            """递归查找具有特定class的节点"""
            if nodes["nodeType"][root_idx] == 1:  # 元素节点
                attrs = nodes["attributes"][root_idx]
                for i in range(0, len(attrs), 2):
                    if to_string(attrs[i]) == "class" and target_class in to_string(attrs[i + 1]):
                        return root_idx
            
            for child_idx in node_children[root_idx]:
                result = find_node_by_class(child_idx, target_class)
                if result is not None:
                    return result
            return None

        # 查找所有论文块
        for node_idx in range(len(nodes["nodeName"])):
            if nodes["nodeType"][node_idx] == 1:  # 元素节点
                attrs = nodes["attributes"][node_idx]
                is_paper_block = False
                # for i in range(0, len(attrs), 2):
                #     if to_string(attrs[i]) == "class" and "gs_r gs_or gs_scl" in to_string(attrs[i + 1]):
                #         is_paper_block = True
                #         break
                # 安全地检查class属性
                for i in range(0, len(attrs), 2):
                    attr_name = to_string(attrs[i])
                    attr_value = to_string(attrs[i + 1])
                    if attr_name == "class" and attr_value and "gs_r gs_or gs_scl" in attr_value:
                        is_paper_block = True
                        break
                
                if is_paper_block:
                    paper_info = {
                        "title": "",
                        "title_link": "",
                        "authors": [],
                        "pdf_link": "",
                        "venue": "", 
                        "citation_count": "",
                        "year": ""
                    }
                    
                    # 查找标题节点 (gs_rt)
                    title_node = find_node_by_class(node_idx, "gs_rt")
                    if title_node:
                        title_links = get_node_links(title_node)
                        if title_links:
                            paper_info["title"] = title_links[0]["text"]
                            paper_info["title_link"] = title_links[0]["href"]
                    
                    # 查找作者节点 (gs_a)
                    author_node = find_node_by_class(node_idx, "gs_a")
                    if author_node:
                        author_text = get_node_text(author_node)
                        # 提取作者链接
                        author_links = get_node_links(author_node)
                        for link in author_links:
                            if link.get("href", "").find("citations?user=") != -1:
                                paper_info["authors"].append({
                                    "name": link["text"].strip(),
                                    "profile_link": link["href"]
                                })
                        
                        # 提取年份和venue信息
                        if author_text:
                            parts = author_text.split(" - ")
                            if len(parts) > 1:
                                # 尝试提取年份
                                import re
                                year_match = re.search(r'\b\d{4}\b', parts[1])
                                if year_match:
                                    paper_info["year"] = year_match.group(0)
                                # 提取venue
                                paper_info["venue"] = parts[-1].strip()
                    
                    # 查找引用数
                    citation_node = find_node_by_class(node_idx, "gs_fl")
                    if citation_node:
                        citation_text = get_node_text(citation_node)
                        citation_match = re.search(r'被引用次数：(\d+)', citation_text)
                        if citation_match:
                            paper_info["citation_count"] = citation_match.group(1)
                    
                    # 查找PDF链接
                    pdf_section = find_node_by_class(node_idx, "gs_ggs gs_fl")
                    if pdf_section:
                        pdf_links = get_node_links(pdf_section)
                        for link in pdf_links:
                            if "[PDF]" in link.get("text", ""):
                                paper_info["pdf_link"] = link["href"]
                                break
                    results.append(paper_info)

        return results
    try:
        # return parse_document(document_idx=0)
        results = parse_document(document_idx=0)
        # 如果结果为空,返回空列表而不是报错
        return results if results else []
    except Exception as e:
        raise ValueError(f"Error encountered when extracting the paper information from the Google Scholar search results: {e}")
def post_process_pdf_link(result_list: list):
    for result in result_list:
        if result["pdf_link"] == "":
            if result["title_link"][-1] == "/":
                result["title_link"] = result["title_link"][:-1]
            if "link.springer.com" in result["title_link"]:
                # https://link.springer.com/article/10.1007/s13042-024-02474-z
                # -> https://link-springer-com.eproxy.lib.hku.hk/content/pdf/10.1007/s13042-024-02474-z.pdf
                assert "article/" in result["title_link"]
                doi = result["title_link"].split("article/")[1]
                result["pdf_link"] = f"https://link-springer-com.eproxy.lib.hku.hk/content/pdf/{doi}.pdf"
            elif "dl.acm.org" in result["title_link"]:
                # https://dl.acm.org/doi/abs/10.1145/3706116
                # -> https://dl-acm-org.eproxy.lib.hku.hk/doi/pdf/10.1145/3706116
                doi = result["title_link"].split("abs/")[1]
                result["pdf_link"] = f"https://dl-acm-org.eproxy.lib.hku.hk/doi/pdf/{doi}"
            elif "ieeexplore.ieee.org" in result["title_link"]:
                # https://ieeexplore.ieee.org/abstract/document/10772600
                # -> https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=10772600
                doi = result["title_link"].split("document/")[1]
                result["pdf_link"] = f"https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber={doi}"
            elif "mdpi.com" in result["title_link"]:
                # https://www.mdpi.com/2076-3417/13/22/12413
                # -> https://www.mdpi.com/2076-3417/13/22/12413/pdf
                result["pdf_link"] = f"{result['title_link']}/pdf"
            elif "nature.com" in result["title_link"]:
                # https://www.nature.com/articles/s41591-024-03233-x
                # -> https://www-nature-com.eproxy.lib.hku.hk/articles/s41591-024-03233-x.pdf
                assert "article/" in result["title_link"]
                doi = result["title_link"].split("article/")[1]
                result["pdf_link"] = f"https://www-nature-com.eproxy.lib.hku.hk/articles/{doi}.pdf"
            elif "science.org" in result["title_link"]:
                # https://www.science.org/doi/abs/10.1126/science.adi2336
                # -> https://www-science-org.eproxy.lib.hku.hk/doi/pdf/10.1126/science.adi2336?download=true
                doi = result["title_link"].split("abs/")[1]
                result["pdf_link"] = f"https://www-science-org.eproxy.lib.hku.hk/doi/pdf/{doi}?download=true"
            else:
                result_list.remove(result) 
    for result in result_list: 
        assert result["pdf_link"] != "", f"The pdf link of the paper {result['title']} is not found"
    return result_list
def wrap_google_scholar_search(web_obs: WebObservation, action_description: str = ""):
    error_prefix = ""
    if web_obs.error:
        error_prefix = get_error_prefix(web_obs.last_browser_action, web_obs.last_browser_action_error)
    cur_url = web_obs.url
    try:
        # cur_axtree_txt = flatten_dom_to_str(
        #     web_obs.dom_object,
        #     extra_properties=web_obs.extra_element_properties,
        #     with_clickable=True,
        #     filter_visible_only=True,
        # )
        # print(web_obs.dom_object)
        cur_axtree_txt = flatten_google_scholar_results(
            web_obs.dom_object,
            extra_properties=web_obs.extra_element_properties
        )
        cur_axtree_txt = post_process_pdf_link(cur_axtree_txt)
    except Exception as e:
        cur_axtree_txt = f'Error encountered when browsing.\nError when trying to process the accessibility tree:{str(e)}'
#     ret_value = f"""\
# {error_prefix}
# {action_description}

# # Current Page URL:
# {cur_url}

# # Current Accessibility Tree:
# {cur_axtree_txt}

# Here is an example with chain of thought of a valid action when clicking on a button:
# "
# In order to accomplish my goal I need to click on the button with bid 12
# ```click("12")```
# "
# """.strip()
    ret_value = json.dumps(cur_axtree_txt, indent=4)
    # ret_value = cur_axtree_txt
    return ret_value
def google_scholar_search(context_variables, env: BrowserEnv, query: str, page: int = 0):
    """
    Performs a Google Scholar search with the given query and page number.

    Args:
        query: The query to search for.
        page: The page number to search for. (default: 0 and start from the 0)
    """
    date_limit = context_variables.get("date_limit", None)
    if date_limit:
        date_limit_str = f"as_ylo=&as_yhi={date_limit.split('-')[0]}"
    else:
        date_limit_str = ""
    try:
        action_str = f"_visit_page('https://scholar.google.com.hk/scholar?start={page*10}&q={quote_plus(query)}&hl=en&as_sdt=0,5{date_limit_str}')"
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_google_scholar_search(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
def mannel_redirect_pdf_link(url: str):
    
    if "link.springer.com" in url:
        # https://link.springer.com/content/pdf/10.1007/s13042-024-02474-z.pdf
        # -> https://link-springer-com.eproxy.lib.hku.hk/content/pdf/10.1007/s13042-024-02474-z.pdf
        doi = url.split("/content/pdf/")[1]
        return f"https://link-springer-com.eproxy.lib.hku.hk/content/pdf/{doi}"
    elif "dl.acm.org" in url:
        # https://dl.acm.org/doi/pdf/10.1145/3534678.3539321
        # -> https://dl-acm-org.eproxy.lib.hku.hk/doi/pdf/10.1145/3534678.3539321
        doi = url.split("/pdf/")[1]
        return f"https://dl-acm-org.eproxy.lib.hku.hk/doi/pdf/{doi}"
    elif "ieeexplore.ieee.org" in url:
        # https://ieeexplore.ieee.org/iel8/5962385/6104215/10767362.pdf
        # -> https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=10767362
        doi = url.split("/")[-1]
        return f"https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber={doi}"
    elif "science.org" in url:
        # https://www.science.org/doi/pdf/10.1126/sciadv.abi7948
        # -> https://www-science-org.eproxy.lib.hku.hk/doi/pdf/10.1126/sciadv.abi7948?download=true
        doi = url.split("doi/pdf/")[1]
        return f"https://www-science-org.eproxy.lib.hku.hk/doi/pdf/{doi}?download=true"
    elif "nature.com" in url:
        # https://www.nature.com/articles/s41586-024-07487-w_reference.pdf
        # -> https://www-nature-com.eproxy.lib.hku.hk/articles/s41586-024-07487-w_reference.pdf
        doi = url.split("/articles/")[1]
        return f"https://www-nature-com.eproxy.lib.hku.hk/articles/{doi}"
    else:
        return url

def download_from_pdf_link(env: BrowserEnv, pdf_link: str, save_name: str):
    try: 
        pdf_link = mannel_redirect_pdf_link(pdf_link)
        cookies = get_all_cookies()
        cookies_dict = {}

        proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据你的代理端口修改
            'https': 'http://127.0.0.1:7890'  # 根据你的代理端口修改
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        for cookie in cookies:
            cookies_dict[cookie['name']] = cookie['value']
        response = requests.get(pdf_link, stream=True, headers=headers,
            cookies=cookies_dict,
            allow_redirects=True, proxies=proxies,verify=False,)
        content_type = response.headers.get('Content-Type', '')
        print(f"Content-Type: {content_type}")

        if 'application/pdf' in content_type:
            # 直接是PDF文件
            pdf_content = response.content
        else:
            # 如果是HTML页面，尝试从中提取PDF URL
            html_content = response.text
            
            # 打印响应内容用于调试
            print("Response content:", html_content[:500])
            
            # 尝试从HTML中提取PDF URL
            pdf_url_match = re.search(r'(https?://[^\s<>"]+?\.pdf)', html_content)
            if pdf_url_match:
                pdf_url = pdf_url_match.group(1)
                print(f"Found PDF URL: {pdf_url}")
                
                # 下载实际的PDF
                pdf_response = requests.get(
                    pdf_url,
                    headers=headers,
                    cookies=cookies_dict,
                    proxies=proxies,
                    verify=False,
                    stream=True
                )
                pdf_content = pdf_response.content
            else:
                # 如果找不到PDF URL，可能需要其他方式处理
                print("No PDF URL found in response")
                return f"No PDF URL found in response"
                # 尝试使用原始响应内容

        # 保存文件
        filename = save_name
        if not filename.endswith('.pdf'):
            filename = filename + '.pdf'
        filepath = env.local_workplace / "papers" / filename
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            f.write(pdf_content)
        docker_path = env.docker_workplace / "papers" / filename
        return f"The pdf file has been downloaded and saved in `{docker_path}`. You should transfer the conversation to the `File Surfer Agent` to open and browse the file."
    except Exception as e:
        return f"Error encountered when downloading the pdf file from {pdf_link}\nError: {e}"

if __name__ == "__main__":
    import json
    env = BrowserEnv(browsergym_eval_env = None, local_root="/home/tjb/llm/agent/Inno-agent", workplace_name="workplace_paper_eval")
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

    res = google_scholar_search({}, env, 'Denoising Diffusion Probabilistic Models', 0)
    # res = visit_url(env, 'https://www.reddit.com/r/ChatGPT/comments/1h5ey4m/chatgpt_helped_me_not_blow_up_on_my_boss/')
    print(res.value)  
    # res = visit_url(env, 'https://dl-acm-org.eproxy.lib.hku.hk/doi/pdf/10.1145/3534678.3539321.pdf')
    # print(res.value)
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