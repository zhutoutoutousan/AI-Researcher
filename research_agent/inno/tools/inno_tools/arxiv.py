import urllib.parse
import feedparser
import time
import requests
import re
import tarfile
import os
def download_arxiv_by_title(title):
    """
    download arxiv paper by title
    
    Args:
        title (str): title of the paper
    
    """
    # 构建API URL
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = urllib.parse.quote(title)
    
    # 设置API参数
    params = {
        'search_query': f'ti:{search_query}',
        'start': 0,
        'max_results': 5,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    
    # 构建完整的查询URL
    query_url = base_url + urllib.parse.urlencode(params)
    
    # 发送请求并解析结果
    response = feedparser.parse(query_url)
    
    # 提取论文信息
    papers = []
    for entry in response.entries:
        paper = {
            'title': entry.title,
            'authors': [author.name for author in entry.authors],
            'published': entry.published,
            'summary': entry.summary,
            'link': entry.link,
            'pdf_link': next(link.href for link in entry.links if link.type == 'application/pdf')
        }
        papers.append(paper)
        
        time.sleep(0.5)
    try: 
        for paper in papers:
            result = download_arxiv_source(paper['link'], paper['title'])
            print(result)
    except Exception as e:
        return f"Download failed with error: {str(e)}"
    
    


def download_arxiv_source(arxiv_url, arxiv_title):
    """
    download arxiv paper source by arxiv url
    
    Args:
        arxiv_url (str): arXiv paper url, e.g. 'http://arxiv.org/abs/2006.11239v2'
    
    Returns:
        success return source content, failed return None
    """
    try:
        # extract paper id from url
        paper_id = re.search(r'abs/([^/]+)', arxiv_url).group(1)
        
        # build source url
        source_url = f'http://arxiv.org/src/{paper_id}'
        
        # send request
        response = requests.get(source_url)
        
        # check status code
        if response.status_code == 200:
            os.makedirs("arxiv_source", exist_ok=True)
            with open(f"arxiv_source/{arxiv_title.replace(' ', '_').lower()}.tar.gz", "wb") as file:
                file.write(response.content)
            try: 
                tex_content = extract_tex_content(f"arxiv_source/{arxiv_title.replace(' ', '_').lower()}.tar.gz")
                os.makedirs("arxiv_papers", exist_ok=True)
                with open(f"arxiv_papers/{arxiv_title.replace(' ', '_').lower()}.tex", "w") as file:
                    file.write(tex_content)
            except Exception as e:
                return f"Download source success, but extract tex file failed with error: {str(e)}"
            return f"Download source success and extract tex file success, saved to `arxiv_papers/{arxiv_title.replace(' ', '_').lower()}.tex`"
        else:
            return f"Download source failed with HTTP status code {response.status_code}"
            
    except Exception as e:
        raise Exception(f"Download failed with error: {str(e)}")


def extract_tex_content(tar_path):
    """
    extract all tex file content from tar.gz file
    
    Args:
        tar_path (str): tar.gz file path
    
    Returns:
        str: all tex file content, format as filename + content
    """
    try:
        all_content = []
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            # get all tex files
            tex_files = [f for f in tar.getmembers() if f.name.endswith('.tex')]
            
            for tex_file in tex_files:
                # extract file content
                f = tar.extractfile(tex_file)
                if f is not None:
                    try:
                        # try to decode with utf-8
                        content = f.read().decode('utf-8')
                    except UnicodeDecodeError:
                        # if utf-8 failed, try latin-1
                        f.seek(0)
                        content = f.read().decode('latin-1')
                    
                    # add filename and content
                    all_content.append(f"\n{'='*50}\n文件名: {tex_file.name}\n{'='*50}\n")
                    all_content.append(content)
                    all_content.append("\n\n")
        
        # join all content
        return "".join(all_content)
    
    except Exception as e:
        return f"Extract failed with error: {str(e)}"
    
if __name__ == "__main__":
    download_arxiv_by_title("Denoising Diffusion Probabilistic Models")
