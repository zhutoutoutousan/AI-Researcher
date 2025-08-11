import arxiv
from typing import Dict, Any, Optional

def get_arxiv_paper_meta(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    get the meta data of the arxiv paper
    
    Args:
        arxiv_id: the arxiv id of the paper, can be the full url or the id (e.g. '2305.02759' or '2305.02759v4')
    
    Returns:
        a dictionary containing the meta information of the paper, return None if the retrieval fails
    """
    try:
        # extract the id from the url
        if 'arxiv.org' in arxiv_id:
            arxiv_id = arxiv_id.split('/')[-1]
        
        # remove the version number
        base_id = arxiv_id.split('v')[0]
        
        # create the client
        client = arxiv.Client()
        
        # search the paper
        search = arxiv.Search(
            id_list=[base_id],
            max_results=1
        )
        
        # get the result
        paper = next(client.results(search))
        # print(paper)
        
        # build the return data
        meta_data = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'abstract': paper.summary,
            'categories': paper.categories,
            'published': paper.published,
            'updated': paper.updated,
            'doi': paper.doi,
            'pdf_url': paper.pdf_url,
            'primary_category': paper.primary_category,
            'comment': paper.comment,
            'journal_ref': paper.journal_ref,
            'version': paper.entry_id.split('v')[-1] if 'v' in paper.entry_id else '1'
        }
        
        return meta_data
        
    except Exception as e:
        print(f"Error fetching arxiv paper meta: {str(e)}")
        return None

# usage example
if __name__ == "__main__":
    # paper_url = "http://arxiv.org/abs/2305.02759v4"
    # meta = get_arxiv_paper_meta(paper_url)
    # print(meta['published'])
    # if meta:
    #     print(f"Title: {meta['title']}")
    #     print(f"Authors: {', '.join(meta['authors'])}")
    #     print(f"Abstract: {meta['abstract'][:200]}...")
    #     print(f"Categories: {meta['categories']}")
    #     print(f"Version: {meta['version']}")
    client = arxiv.Client()
        
    # search the paper
    search = arxiv.Search(
        query="Denoising Diffusion Probabilistic Models proceedings.neurips.cc",
        max_results=3
    )
    for result in client.results(search):
        print(result.title)