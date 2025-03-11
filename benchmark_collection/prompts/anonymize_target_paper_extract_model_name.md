Given a research paper's title and abstract, extract the name of the novel model/method introduced in the paper:

1. Look for phrases that signal a new model introduction, such as:
   - "we propose/present/introduce"
   - "our model/method/approach"
   - "called/named"
   - Model name followed by model architecture details

2. Return format:
   - If a proposed model name is found: Return only the model name
   - If you find both abbreviation and full name for the model, format them into "abbreviation, full name"
   - If no clear model name is found: Return "NO MODEL NAME FOUND"
   - You should strictly follow the requirement, and output without any other words

3. Focus only on the main proposed model:
   - Ignore baseline models
   - Ignore models from referenced papers
   - Ignore general model categories/types

Input:
- Paper Title: {paper_title}
- Paper Content: {paper_content}