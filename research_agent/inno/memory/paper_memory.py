import pandas as pd
from typing import List, Dict
from research_agent.inno.memory.rag_memory import Memory, Reranker
import json
import math
import os
from litellm import completion
from research_agent.inno.memory.utils import chunking_by_token_size

class PaperMemory(Memory):
    def __init__(
        self,
        project_path: str,
        db_name: str = '.paper_table',
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
        self.collection_name = 'paper_memory'

    def add_paper_content(self, paper_content: str, batch_size: int = 100, collection = None):
        assert collection is not None, "Collection is required. Should be the path of the paper."
        queries = []
        content_chunks = chunking_by_token_size(paper_content, max_token_size=4096)

        idx_list = ["chunk_" + str(chunk['chunk_order_index']) for chunk in content_chunks]
        for chunk in content_chunks:
            query = {
                'query': chunk['content'],
                'response': chunk['content']
            }
            queries.append(query)
        
        # self.add_query(queries, collection=collection)
        print(f'Adding {len(queries)} queries to {collection} with batch size {batch_size}')
        num_batches = math.ceil(len(queries) / batch_size)
    
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries))
            batch_queries = queries[start_idx:end_idx]
            batch_idx = idx_list[start_idx:end_idx]
            
            # Add the current batch of queries
            self.add_query(batch_queries, collection=collection, idx=batch_idx)
            
            print(f"Batch {i+1}/{num_batches} added")

    def query_paper_content(
        self, 
        query_text: str, 
        collection: str = None, 
        n_results: int = 5
    ) -> List[str]:
        """
        Query the table and return the results
        """
        assert collection is not None, "Collection is required. Should be the path of the paper."
        results = self.query([query_text], collection=collection, n_results=n_results)
        
        metadata_results = results['metadatas'][0]
        
        results = [item['response'] for item in metadata_results]
        return results

    def peek_table(self, collection: str = None, n_results: int = 20) -> pd.DataFrame:
        """
        Peek at the data in the table
        """
        assert collection is not None, "Collection is required. Should be the path of the paper."
        raw_results = self.peek(collection=collection, n_results=n_results)
        results = [item['response'] for item in raw_results['metadatas']]
        return results