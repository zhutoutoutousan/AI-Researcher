from research_agent.inno.memory.tool_memory import ToolMemory, ToolReranker
import os
from research_agent.inno.io_utils import get_file_md5
import pandas as pd
from research_agent.inno.registry import register_tool

@register_tool("get_api_doc")
def get_api_doc(query_text: str) -> str:
    """
    Retrieve satisfied tool documents based on the query text.
    Args:
        query_text: A query or request from users and you need to find the satisfied tool documents based on the query text.
    Returns:
        A string representation of the reranked results.
    """
    tool_memory = ToolMemory(project_path = './code_db', db_name = ".tool_table", platform='OpenAI', api_key=os.getenv("OPENAI_API_KEY"), embedding_model='text-embedding-3-small')
    tool_reranker = ToolReranker(model="gpt-4o-2024-08-06")
    tool_path = "./tool_docs.csv"
    code_id = get_file_md5(tool_path)
    # print(code_id)
    tool_memory.collection_name = tool_memory.collection_name + f"_{code_id}"
    if tool_memory.count() == 0:
        tool_memory.add_dataframe(pd.read_csv(tool_path), batch_size=100)
    res_df = tool_memory.query_table(query_text, n_results=20)
    try:
        reranked_df = tool_reranker.rerank(query_text, res_df)
    except Exception as e:
        return res_df
    wrapped_res = \
f"""
The referenced tool documentation is:
API Name: {reranked_df[0]['API_Name'].values[0]}
API Description: {reranked_df[0]['API_Description'].values[0]}
API Details: {reranked_df[0]['API_Details'].values[0]}
Required API Key: {reranked_df[0]['Required_API_Key'].values[0]}
Platform: {reranked_df[0]['Platform']}
"""
    return wrapped_res
    
