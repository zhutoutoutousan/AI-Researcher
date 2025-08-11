This directory includes the scripts to create the innovation benchmark dataset. Use it in 3 steps:
- Create a list of paper titles or keywords in the `papers_to_search` file.
- Run `python 0_crawl_paper.py` to collect papers and the related meta data.
- Run `1_create_inno_graph.py` to get the final innovation dataset.

Here are some important notes:
- If you want to collect papers related to some keywords, change exact_match to False in `0_crawl_paper.py`.
- You need to set your OpenAI key globally, or edit `utils/openai_utils.py`.
- Your final results is stored at `innovation_graph/innovation_graph_final.json`.
- The innovation dataset we used in our evaluation is `merged_papers_with_fields.json`. It has gone through some additional manual revisement.
