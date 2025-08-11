import os
import json
import asyncio
import logging
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_collection.utils.openai_utils import GPTClient
from paper_agent.section_composer import SectionComposer, setup_logging

class RelatedWorkComposer(SectionComposer):
    def __init__(self, research_field: str, structure_iterations: int = 3):
        super().__init__(research_field, "related_work", structure_iterations)

    async def generate_or_revise_structure(self, content: str, current_structure: str, iteration: int) -> str:
        prompt = f"""Based on the given content, generate or revise the related work structure, using latex format.
Current iteration: {iteration}/{self.structure_iterations}

Current structure (if exists):
{current_structure}

Content to analyze:
{content}

Guidelines for structure generation:
1. SECTION ORGANIZATION:
   - Main section should be "Related Work" (with latex command \section{{Related Work}})
   - Use 2-3 subsections for different research directions/categories
   - Group related papers logically under each subsection
   - Ensure proper flow from foundational to advanced topics

2. SECTION HIERARCHY:
   - Use subsections for major research directions
   - Group papers by methodology or approach
   - Maintain chronological order within groups when relevant

3. REQUIRED COMMENTS:
   Add latex comments (start with %) under each section/subsection to explain:

   For Related Work section:
   - Overview of the research landscape
   - Key research directions and their relationships
   - Evolution of the field
   - Connection to the proposed work

   For each subsection:
   - Key papers and their contributions
   - Technical approaches and methodologies
   - Current limitations and challenges
   - Relevance to the proposed work

4. STRUCTURE FORMAT:
   \section{{Related Work}}

   \subsection{{Related Research Direction 1}}
   % [Key approaches and limitations]
   % [Connection to modern approaches]
   
   \subsection{{Related Research Direction 2}}
   % [Key architectures and innovations]
   % [Remaining challenges]
   
   \subsection{{Related Research Direction 3}}
   % [Recent innovations]
   % [Future directions]

   % Do not use \subsubsection

Output only the LaTeX structure with comments as specified above."""

        return await self.gpt_client.chat(prompt=prompt)

    async def detailize_subsection(self, structure: str, current_text: str, content: str, subsection: str) -> str:
        writing_template = self.get_random_template()
        
        prompt = f"""Write or revise the following subsection of the related work:
\subsection{{{subsection}}}

CURRENT TEXT (if any):
{current_text}

STRUCTURE INFORMATION:
{structure}

NEW CONTENT TO INCORPORATE:
{content}

REFERENCE WRITING TEMPLATE:
{writing_template}

Requirements for related work writing:
1. PAPER ORGANIZATION:
   - Group papers by methodology/approach
   - Present chronological development
   - Highlight key contributions
   - Show evolution of ideas

2. CRITICAL ANALYSIS:
   - Compare different approaches
   - Identify strengths and limitations
   - Discuss technical innovations
   - Note remaining challenges

3. WRITING STYLE:
   - Use clear transitions between papers
   - Maintain objective tone
   - Balance detail level
   - Ensure technical accuracy

4. CITATIONS:
   - Find as many references as you can from the new content
   - Don't cite papers that do not exist
   - Use proper citation format
   - Group related citations
   - Cite seminal works
   - Include recent developments

5. TECHNICAL CONTENT:
   - Focus on methodological aspects
   - Highlight key innovations
   - Discuss technical limitations
   - Connect to your work

Output the detailed LaTeX text for this subsection only."""

        return await self.gpt_client.chat(prompt=prompt)

    async def final_writing_checklist(self, related_work_text: str) -> str:
        prompt = f"""Review and revise the related work section following these academic writing guidelines:

Current related work text:
{related_work_text}

CHECKLIST FOR REVISION:

1. \section{{Related Work}} is directly followed by the subsections, without section-level text.
2. Each subsection first discusses the related works, and then shortly discusses the relative contribution of our work. No other content.
3. Discussion on related works should have at least two times the length of the discussion on the contribution of our work.
4. Each subsection should cite at least 4 papers, and at most 10 papers.
5. Each subsection should contains one to two paragraphs.
6. No equations
7. Properly cite the related works using the \cite{{}} command
8. List all reference papers with all publication information at the end, in the bibtex format.

Output the revised related work section incorporating all these improvements. Reply with LaTeX code only."""

        return await self.gpt_client.chat(prompt=prompt)

    def read_related_papers(self, papers_dir):
        """Read all related papers from the papers directory"""
        papers_content = []
        for filename in os.listdir(papers_dir):
            if filename.endswith('.txt') or filename.endswith('.json'):
                try:
                    with open(os.path.join(papers_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                        papers_content.append({
                            'filename': filename,
                            'content': content
                        })
                except Exception as e:
                    logging.error(f"Error reading paper file {filename}: {str(e)}")
        return papers_content
    
    async def compose_section(self, agent_dir: str, papers_dir: str, benchmark_path: str, target_paper: str) -> str:
        checkpoint_dir = self.get_checkpoint_path(target_paper)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Focus on literature review related agent files
        agent_files = [
            'prepare_agent.json',
            'survey_agent.json',

            # 'coding_plan_agent.json',
            # 'machine_learning_agent.json',
            # 'judge_agent.json',
            # 'machine_learning_agent_iter_submit.json',
            # 'experiment_analysis_agent_iter_refine_1.json',
            # 'machine_learning_agent_iter_refine_1.json',
        ]
        
        # Read related papers
        related_papers = self.read_related_papers(papers_dir)
        logging.info(f"Found {len(related_papers)} related papers in {papers_dir}")

        # Step 1: Iterative structure generation
        structure = ""
        structure_checkpoint = self.load_checkpoint(target_paper, "structure")
        
        if structure_checkpoint:
            structure = structure_checkpoint["final_structure"]
            logging.info("Loaded structure from checkpoint")
        else:
            for iteration in range(self.structure_iterations):
                logging.info(f"Structure iteration {iteration + 1}/{self.structure_iterations}")
                
                # Process agent files for literature information
                for idx, agent_file in enumerate(tqdm(agent_files, desc="Processing agent files")):
                    with open(os.path.join(agent_dir, agent_file), 'r') as f:
                        content = json.load(f)
                    structure = await self.generate_or_revise_structure(
                        json.dumps(content, indent=2), structure, iteration + 1)
                
                self.write_temp_log(structure, f"iteration_{iteration+1}_final")
            
            self.save_checkpoint(target_paper, "structure", {
                "final_structure": structure
            })

        # Step 2: Detailize subsections
        subsections = [line.split('{')[1].split('}')[0] 
                    for line in structure.split('\n') 
                    if line.strip().startswith('\\subsection')]
        
        subsection_contents = {}
        subsection_checkpoint = self.load_checkpoint(target_paper, "subsections")
        
        if subsection_checkpoint:
            subsection_contents = subsection_checkpoint
            logging.info("Loaded subsection contents from checkpoint")
        else:
            for subsection_id, subsection in enumerate(tqdm(subsections, desc="Detailizing subsections")):
                related_work_part = ''
                
                # First process agent contents
                for i, agent_file in enumerate(agent_files):
                    with open(os.path.join(agent_dir, agent_file), 'r') as f:
                        content = json.load(f)
                    related_work_part = await self.detailize_subsection(
                        structure, related_work_part, json.dumps(content, indent=2), subsection)
                    
                    self.write_temp_log(
                        related_work_part,
                        f"subsection_{subsection_id}_agent_{i}"
                    )
                
                # Then process related papers
                for i, paper in enumerate(tqdm(related_papers, desc=f"Processing related papers for subsection {subsection}")):
                    related_work_part = await self.detailize_subsection(
                        structure, related_work_part, paper['content'], subsection)
                    
                    self.write_temp_log(
                        related_work_part,
                        f"subsection_{subsection_id}_paper_{i}"
                    )
                    
                    subsection_contents[subsection] = related_work_part
                    self.save_checkpoint(target_paper, "subsections", subsection_contents)


        # Step 3: Fuse all subsections
        self.write_temp_log(
            json.dumps(subsection_contents, indent=2),
            "pre_fusion_subsections"
        )
        
        fused_related_work = await self.fuse_subsections(structure, subsection_contents)
        self.write_temp_log(fused_related_work, "post_fusion_related_work")

        # Step 4: Final writing checklist
        final_related_work = await self.final_writing_checklist(fused_related_work)
        self.write_temp_log(final_related_work, "post_checklist_related_work")

        # Save final output
        output_dir = f"{self.research_field}/target_sections/{self.normalize_title(target_paper)}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "related_work.tex")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_related_work)
        logging.info(f"Saved final related work to {output_path}")

        return final_related_work

async def related_work_composing(research_field: str, instance_id: str):
    setup_logging(research_field)
    
    composer = RelatedWorkComposer(research_field=research_field, structure_iterations=1)
    
    # proj_dir = f'/data2/tjb_share/{research_field}/{instance_id}/'
    proj_dir = f'./paper_agent/{research_field}/{instance_id}/'
    # target_paper = "Knowledge Graph Self-Supervised Rationalization for Recommendation"
    # target_paper = 'Heterogeneous Graph Contrastive Learning for Recommendation'
    cache_dirs = [d for d in os.listdir(proj_dir) if d.startswith('cache_')]
    if not cache_dirs:
        raise ValueError("No cache directory found")
    agent_dir = os.path.join(proj_dir, cache_dirs[-1], 'agents')
    
    # benchmark_path = f'/data2/tjb/Inno-agent/benchmark/final/{research_field}/{instance_id}.json'
    benchmark_path = f'./benchmark/final/{research_field}/{instance_id}.json'
    papers_dir = os.path.join(proj_dir, 'workplace', 'papers')
    
    try:
        related_work = await composer.compose_section(
            agent_dir, papers_dir, benchmark_path, instance_id)
        logging.info("Related work composition completed")
    except Exception as e:
        logging.error(f"Error during related work composition: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(related_work_composing())