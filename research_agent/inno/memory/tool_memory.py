import pandas as pd
from typing import List, Dict
from research_agent.inno.memory.rag_memory import Memory, Reranker
import json
import math
import os
from litellm import completion
"""
Category | Tool_Name | Tool_Description | API_Name | API_Description | Method | API_Details | Required_API_Key | Platform
"""
class ToolMemory(Memory):
    def __init__(
        self,
        project_path: str,
        db_name: str = '.tool_table',
        platform: str = 'OpenAI',
        api_key: str = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        super().__init__(
            project_path=project_path,
            db_name=db_name,
            platform=platform,
            api_key=api_key,
            embedding_model=embedding_model
        )
        self.collection_name = 'tool_memory'

    def add_dataframe(self, df: pd.DataFrame, collection: str = None, batch_size: int = 100):
        if not collection:
            collection = self.collection_name
        queries = []
        for idx, row in df.iterrows():
            query = {
                'query': ' '.join(row.astype(str)),
                'response': row.to_json()
            }
            queries.append(query)
        
        # self.add_query(queries, collection=collection)
        print(f'Adding {len(queries)} queries to {collection} with batch size {batch_size}')
        num_batches = math.ceil(len(queries) / batch_size)
    
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries))
            batch_queries = queries[start_idx:end_idx]
            
            # Add the current batch of queries
            self.add_query(batch_queries, collection=collection)
            
            print(f"Batch {i+1}/{num_batches} added")

    def query_table(
        self, 
        query_text: str, 
        collection: str = None, 
        n_results: int = 5
    ) -> pd.DataFrame:
        """
        Query the table and return the results
        """
        if not collection:
            collection = self.collection_name
        results = self.query([query_text], collection=collection, n_results=n_results)
        
        metadata_results = results['metadatas'][0]
        
        df_results = pd.DataFrame([json.loads(item['response']) for item in metadata_results])
        return df_results

    def peek_table(self, collection: str = None, n_results: int = 20) -> pd.DataFrame:
        """
        Peek at the data in the table
        """
        if not collection:
            collection = self.collection_name
        results = self.peek(collection=collection, n_results=n_results)
        df_results = pd.DataFrame([json.loads(item['response']) for item in results['metadatas']])
        return df_results

class ToolReranker(Reranker):
    def rerank(self, query_text: str, query_df: pd.DataFrame) -> str:
        system_prompt = \
        """
        You are a helpful assistant that reranks the given API table based on the query.
        You should select the top 5 APIs to answer the query, by giving the `Tool_Name` and `API_Name` of the APIs in the format: `Tool_Name1|API_Name1,Tool_Name2|API_Name2,Tool_Name3|API_Name3,Tool_Name4|API_Name4,Tool_Name5|API_Name5`.
        You can only select APIs I give you.
        Directly give the answer without any other words.
        """
        # Use the DataFrame's to_dict method to convert all rows to a list of dictionaries
        # print('query_df', query_df)
        api_data = query_df.to_dict(orient='records')
        
        # Use a list comprehension and f-string to format each API's data
        api_prompts = [f"\n\nAPI {i+1}:\n{api}" for i, api in enumerate(api_data)]
        
        # add the query text to the prompt
        prompt = ''.join(api_prompts)
        prompt = f"The query is: {query_text}\n\n{prompt}"
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        create_params = {
            "model": self.model,
            "messages": message,
            "stream": False,
        }
        response = completion(**create_params).choices[0].message.content
        # print(response)
        try:
            # tool_api_names = [idx.strip() for idx in response.split(',')]
            # return [query_df.loc[query_df['API_Name'] == tool_api.split('|')[1] & query_df['Tool_Name'] == tool_api.split('|')[0]] for tool_api in tool_api_names]
            tool_api_names = [idx.strip() for idx in response.split(',')]
            result = []
            for tool_api in tool_api_names:
                tool_name, api_name = tool_api.split('|')
                matched_rows = query_df[(query_df['API_Name'] == api_name) & (query_df['Tool_Name'] == tool_name)]
                if not matched_rows.empty:
                    result.append(matched_rows)
            return result
        except Exception as e:
            # If the LLM's output cannot be parsed, return the original order
            raise ValueError(f"LLM output cannot be parsed: {response}, {e}")