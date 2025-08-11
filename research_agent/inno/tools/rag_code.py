from research_agent.inno.memory.code_memory import CodeMemory, CodeReranker
import os
from research_agent.inno.environment.docker_env import DockerEnv
from research_agent.inno.io_utils import compress_folder, get_file_md5
from research_agent.inno.registry import register_tool

@register_tool("code_rag")
def code_rag(query_text: str, env: DockerEnv) -> str:
    """
    Retrieve information from a code directory. Use this function when there is a need to search for information in the codebase.
    Args:
        query_text: Anything you want to search in the code directory, like a function name, a class name, a variable name, etc.
    Returns:
        A string representation of the reranked results.
    """
    code_memory = CodeMemory(project_path = './code_db', platform='OpenAI', api_key=os.getenv("OPENAI_API_KEY"),embedding_model='text-embedding-3-small')
    code_reranker = CodeReranker(model="gpt-4o-2024-08-06")
    code_path = f"{env.local_workplace}/metachain"
    compress_folder(code_path, f"{env.local_workplace}/", "metachain.zip")
    code_id = get_file_md5(f"{env.local_workplace}/metachain.zip")
    code_memory.collection_name = code_memory.collection_name + f"_{code_id}"

    if code_memory.count() == 0:
        code_memory.add_code_files(f"{env.local_workplace}/metachain", exclude_prefix=['__pycache__', 'code_db', '.git'])

    query_results = code_memory.query_code(query_text, n_results=20)
    reranked_results = code_reranker.rerank(query_text, query_results)
    return reranked_results
    
    

