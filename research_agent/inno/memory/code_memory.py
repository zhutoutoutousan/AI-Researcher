import os
from typing import List, Dict
from research_agent.inno.memory.rag_memory import Memory, Reranker
from litellm import completion
import re

class CodeMemory(Memory):
    def __init__(self, project_path: str, db_name: str = '.sa', platform: str = 'OpenAI', api_key: str = None, embedding_model: str = "text-embedding-ada-002"):
        super().__init__(project_path, db_name, platform, api_key, embedding_model)
        self.collection_name = 'code_memory'

    def add_code_files(self, directory: str, exclude_prefix: List[str] = ["workplace_"]):
        """
        Add all code files in the specified directory to the memory.
        
        Args:
            directory (str): The directory path containing the code files to add.
        """
        code_files = []
        for root, _, files in os.walk(directory):
            root_name = str(root)
            if any(prefix in root_name for prefix in exclude_prefix):
                continue
            for file in files:
                
                if file.endswith(('.py', '.js', '.java', '.cpp', '.h', '.c', '.html', '.css')):  # add more file types if needed
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    code_files.append({
                        "query": f"File: {file_path}\n\nContent:\n{content}",
                        "response": f"This is the content of file {file_path}"
                    })
        self.add_query(code_files, self.collection_name)

    def query_code(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """
        Query the code memory.
        
        Args:
            query_text (str): The query text
            n_results (int): The number of results to return
        
        Returns:
            List[Dict]: The query results list
        """
        results = self.query([query_text], self.collection_name, n_results)
        return [
            {
                "file": doc.split('\n')[0].replace("File: ", ""),
                "content": '\n'.join(doc.split('\n')[3:]),
                "metadata": metadata
            }
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0])
        ]
    
class CodeReranker(Reranker):
    def __init__(self, model: str) -> None:
        super().__init__(model)
    def wrap_query_results(self, query_results: List[Dict]) -> str:
        wrapped_query_results = ""
        for result in query_results:
            wrapped_query_results += f"File: {result['file']}\n"
            wrapped_query_results += f"Content: {result['content'][:300]}...\n"
            wrapped_query_results += "---"
        return wrapped_query_results
    def wrap_reranked_results(self, reranked_paths: List[str]) -> str:
        wrapped_reranked_results = "[Referenced code files]:"
        for path in reranked_paths:
            wrapped_reranked_results += f"Code path: {path}\n"
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                wrapped_reranked_results += f"Code content:\n{content}\n"
            except Exception as e:
                wrapped_reranked_results += f"Error reading file: {str(e)}\n"
            wrapped_reranked_results += "---\n"
        return wrapped_reranked_results
    def parse_results(self, reranked_results: str) -> List[str]:
        lines = reranked_results.strip().split('\n')
    
        # get the last 5 lines
        last_lines = lines[-5:]
        
        # remove the number and dot at the beginning of each line
        cleaned_lines = [re.sub(r'^\d+\.\s*', '', line.strip()) for line in last_lines]
        unique_lines = list(dict.fromkeys(cleaned_lines))
        
        return unique_lines
    def rerank(self, query_text: str, query_results: List[Dict]) -> List[Dict]:
        system_prompt = \
        """
        You are a helpful assistant that reranks the given code files (containing the path of files and Overview of the content of files) based on the query.
        You should rerank the code files based on the query, and the most relevant code files should be ranked on the top. 
        You should select the top 5 code files to answer the query, by giving the file path of the code files.

        Example: 
        [Query]: "The definition of 'BaseAgent'"
        [Code files]: 
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/__init__.py
        Content: from .ABCAgent import ABCAgent
        from .BaseAgent import BaseAgent
        from .ManagerAgent import ManagerAge...
        ---
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/__init__.py
        Content: from .ABCAgent import ABCAgent
        from .BaseAgent import BaseAgent
        from .ManagerAgent import ManagerAge...
        ---
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/__init__.py
        Content: from .ABCAgent import ABCAgent
        from .BaseAgent import BaseAgent
        from .ManagerAgent import ManagerAge...
        ---
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/__init__.py
        Content: from .ABCAgent import ABCAgent
        from .BaseAgent import BaseAgent
        from .ManagerAgent import ManagerAge...
        ---
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agent_prompts/__init__.py
        Content: from .BasePrompt import BasePromptGen, ManagerPromptGen, PromptGen
        ...
        ---
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agent_prompts/__init__.py
        Content: from .BasePrompt import BasePromptGen, ManagerPromptGen, PromptGen
        ...
        ---
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agent_prompts/__init__.py
        Content: from .BasePrompt import BasePromptGen, ManagerPromptGen, PromptGen
        ...
        ---
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agent_prompts/__init__.py
        Content: from .BasePrompt import BasePromptGen, ManagerPromptGen, PromptGen
        ...
        ---
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/BaseAgent.py
        Content: from typing import List

        from sa.actions import BaseAction, FinishAct, ThinkAct, PlanAct
        from sa.age...
        ---
        File: /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/BaseAgent.py
        Content: from typing import List

        from sa.actions import BaseAction, FinishAct, ThinkAct, PlanAct
        from sa.age...
        ---
        [Reranked 5 code files]:
        1. /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/BaseAgent.py
        2. /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/__init__.py
        3. /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/ABCAgent.py
        4. /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/ManagerAgent.py
        5. /Users/tangjiabin/Documents/reasoning/SelfAgent/sa/agents/AgentLogger.py
        """
        wrapped_query_results = self.wrap_query_results(query_results)
        user_prompt = \
        """
        [Query]: \n{query_text}
        [Code files]: \n{query_results}
        [Reranked 5 code files]:
        """.format(query_text=query_text, query_results=wrapped_query_results)
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        create_params = {
            "model": self.model,
            "messages": chat_history,
            "stream": False,
        }
        response = completion(**create_params)
        reranked_results = self.parse_results(response.choices[0].message.content)
        reranked_results = self.wrap_reranked_results(reranked_results)
        return reranked_results
