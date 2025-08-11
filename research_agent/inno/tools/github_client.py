import os
import requests
import json
from typing import Optional, Dict, List
import time
class GitHubClient:
    """GitHub operation client"""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub client
        
        Args:
            token: GitHub Personal Access Token, if None, try to get from environment variable
        """
        self.token = token or os.getenv('GITHUB_AI_TOKEN')
        if not self.token:
            raise ValueError("GitHub Token is required, please provide it via the token parameter or set the GITHUB_AI_TOKEN environment variable.")
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        })
        self.api_base = 'https://api.github.com'
    
    def check_auth(self) -> dict:
        """
        Check if the token is valid
        """
        try:
            response = self.session.get(f'{self.api_base}/user')
            response.raise_for_status()
            return {'status': 0, 'message': 'Authentication successful', 'user': response.json()}
        except Exception as e:
            return {'status': -1, 'message': f'Authentication failed: {str(e)}'}
    
    def create_pull_request(self, repo: str, title: str, body: str, head: str, base: str) -> dict:
        """
        Create a Pull Request
        
        Args:
            repo: The full name of the repository (e.g., 'owner/repo')
            title: The PR title
            body: The PR description
            head: The source branch
            base: The target branch
        """
        try:
            url = f'{self.api_base}/repos/{repo}/pulls'
            data = {
                'title': title,
                'body': body,
                'head': head,
                'base': base
            }
            response = self.session.post(url, json=data)
            response.raise_for_status()
            pr_data = response.json()
            return {
                'status': 0,
                'message': f'PR created successfully: {pr_data["html_url"]}',
                'pr_url': pr_data['html_url']
            }
        except Exception as e:
            return {'status': -1, 'message': f'PR creation failed: {str(e)}'}



class GitHubSearcher:
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub searcher
        
        Args:
            token: GitHub Personal Access Token, optional
        """
        self.session = requests.Session()
        if token:
            self.session.headers.update({
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json'
            })
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    # def search_code(self, 
    #                 repo_owner: str, 
    #                 repo_name: str, 
    #                 query: str, 
    #                 language: Optional[str] = None,
    #                 per_page: int = 1) -> Dict:
    #     """
    #     Search code in a specific repository
        
    #     Args:
    #         repo_owner: The owner of the repository
    #         repo_name: The name of the repository
    #         query: The search keyword
    #         language: The programming language filter, optional
    #         per_page: The number of results per page
            
    #     Returns:
    #         dict: The search results
    #     """
    #     # Modify the search URL
    #     base_url = "https://api.github.com/search/code"  # Modify here
        
    #     # Build the query parameters
    #     q = f"repo:{repo_owner}/{repo_name} {query}"
    #     if language:
    #         q += f" language:{language}"
        
    #     params = {
    #         'q': q,
    #         'per_page': per_page  # add this parameter
    #     }
        
    #     try:
    #         response = self.session.get(base_url, params=params)
    #         response.raise_for_status()  # Check if the request is successful
            
    #         # Handle rate limiting
    #         if 'X-RateLimit-Remaining' in response.headers:
    #             remaining = int(response.headers['X-RateLimit-Remaining'])
    #             if remaining < 10:  # If the remaining requests are less, pause for a while
    #                 reset_time = int(response.headers['X-RateLimit-Reset'])
    #                 sleep_time = reset_time - time.time()
    #                 if sleep_time > 0:
    #                     time.sleep(sleep_time)
            
    #         return response.json()
            
    #     except requests.exceptions.RequestException as e:
    #         return {
    #             'status': 'error',
    #             'message': f"Request failed: {str(e)}",
    #             'items': []
    #         }
    
    # def get_file_content(self, file_url: str) -> str:
    #     """
    #     Get the content of a file
        
    #     Args:
    #         file_url: The URL of the file
            
    #     Returns:
    #         str: The content of the file
    #     """
    #     try:
    #         response = self.session.get(file_url)
    #         response.raise_for_status()
    #         return response.json()['content']
            
    #     except requests.exceptions.RequestException as e:
    #         return f"Failed to get file content: {str(e)}"
    def search_code(self, 
                    repo_owner: str, 
                    repo_name: str, 
                    query: str, 
                    language: Optional[str] = None,
                    per_page: int = 5, 
                    page: int = 1) -> Dict:
        """搜索代码"""
        base_url = "https://api.github.com/search/code"
        
        # 构建查询
        q = f"repo:{repo_owner}/{repo_name} {query}"
        if language:
            q += f" language:{language}"
        
        params = {
            'q': q,
            'per_page': min(per_page, 100),  # 确保不超过最大限制
            'page': page
        }
        
        try:
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            
            # 处理速率限制
            self._handle_rate_limit(response.headers)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f"Request failed: {str(e)}",
                'items': []
            }
    
    def get_contents_batch(self, items: List[Dict]) -> List[Dict]:
        """批量获取文件内容"""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_item = {
                executor.submit(self._get_single_content, item): item 
                for item in items
            }
            
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error getting content: {str(e)}")
        
        return results
    
    def _get_single_content(self, item: Dict) -> Dict:
        """获取单个文件的内容"""
        try:
            response = self.session.get(item['url'])
            response.raise_for_status()
            self._handle_rate_limit(response.headers)
            
            file_data = response.json()
            if 'download_url' in file_data:
                content_response = self.session.get(file_data['download_url'])
                content_response.raise_for_status()
                content = content_response.text
            else:
                content = file_data.get('content', '')
                if content:
                    import base64
                    content = base64.b64decode(content).decode('utf-8')
            
            return {
                'name': item['name'],
                'path': item['path'],
                'url': item['html_url'],
                'repository': item['repository']['full_name'],
                'content': content
            }
        except Exception as e:
            return {
                'name': item['name'],
                'path': item['path'],
                'url': item['html_url'],
                'repository': item['repository']['full_name'],
                'content': f"Error: {str(e)}"
            }
    
    def _handle_rate_limit(self, headers: Dict):
        """处理 API 速率限制"""
        if 'X-RateLimit-Remaining' in headers:
            remaining = int(headers['X-RateLimit-Remaining'])
            if remaining < 10:
                reset_time = int(headers['X-RateLimit-Reset'])
                sleep_time = reset_time - time.time()
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 5))  # 最多等待5秒