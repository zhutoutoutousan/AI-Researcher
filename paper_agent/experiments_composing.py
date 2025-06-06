import os
import json
import asyncio
import logging
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_collection.utils.openai_utils import GPTClient
from paper_agent.section_composer import SectionComposer, setup_logging

class ExperimentsComposer(SectionComposer):
    def __init__(self, research_field: str, structure_iterations: int = 3, gpt_model='gpt-4o-mini-2024-07-18'):
        super().__init__(research_field, "experiments", structure_iterations)

    def read_project_structure(self, project_dir):
        """Read entire project directory structure and code files"""
        dir_tree = []
        code_contents = []
        
        for root, dirs, files in os.walk(project_dir):
            # Skip system directories
            dirs[:] = [d for d in dirs if not d.startswith(('__', '.', 'cache'))]
            
            rel_path = os.path.relpath(root, project_dir)
            if rel_path == '.':
                rel_path = ''
                
            current_dir = []
            for f in files:
                if f.endswith('.py'):
                    try:
                        filepath = os.path.join(root, f)
                        with open(filepath, 'r', encoding='utf-8') as file:
                            content = file.read()
                            code_contents.append({
                                'path': os.path.join(rel_path, f),
                                'content': content
                            })
                    except Exception as e:
                        logging.error(f"Error reading file {f}: {str(e)}")
                current_dir.append(f)
            
            if current_dir or dirs:
                dir_tree.append({
                    'path': rel_path,
                    'files': current_dir,
                    'dirs': dirs
                })
                
        return dir_tree, code_contents

    def generate_project_summary(self, dir_tree, code_contents):
        """Generate a structured summary of project files and contents"""
        summary = {
            'directory_structure': dir_tree,
            'code_files': [{'path': code['path']} for code in code_contents]#,
            # 'experiment_scripts': [{'path': script['path']} for script in scripts]
        }
        return json.dumps(summary, indent=2)

    async def generate_or_revise_structure(self, content: str, current_structure: str, iteration: int) -> str:
        prompt = f"""Based on the given content, generate or revise the experiments section structure, using latex format.
Current iteration: {iteration}/{self.structure_iterations}

Current structure (if exists):
{current_structure}

Content to analyze:
{content}

Guidelines for structure generation:
1. SECTION ORGANIZATION:
   The experiments section should be organized into two main parts:
   a) Experimental Settings (as a comprehensive subsection with multiple subsubsections)
   b) Multiple Results subsections (each focusing on different experimental aspects)

2. SECTION HIERARCHY:
   \section{{Experiments}}
   \subsection{{Experimental Settings}}
   \subsubsection{{Datasets and Preprocessing}}
   \subsubsection{{Evaluation Metrics}}
   \subsubsection{{Baselines}}
   \subsubsection{{Implementation Details}}
   
   \subsection{{Main Performance Comparison}}
   \subsection{{Ablation Studies}}
   [Additional experimental subsections based on project files...]

3. REQUIRED COMMENTS:
   Add latex comments (%) to explain:
   a. For each experiment, the comment should include the following information:
        - Purpose and methodology of each experiment
        - Experimental results of this experiment (Present all real results you found in the input files. Using the table format as much as you can)
        - Summary of insights and findings from the results
        - Connections to research claims
   b. For datasets subsubsection, the comment should list the datasets used in experiments
   c. For baselines subsubsection, the comment should list the baselines used in experiments
   d. For implementation details, list relevant information you find in the comment for this subsection
   e. 
   Note that all these information should be in latex comments.

4. EXPERIMENT DETECTION:
   - Analyze project files and directory structure to find experimental scripts (e.g. run_*.py files)
   - Read agent files to find the experiment descriptions, experimental results and analysis
   - Add new subsections for uncovered experiment types
   - Based on description and result summary in each experiment's comments, deduplicate subsections for the same experiment

5. OTHER REQUIREMENTS:
   - There are only two types of subsections: experimental settings and individual experiments
   - Each experiment subsection is to test the performance of the proposed method from certain dimension. Each experiment subsection should include the purpose and methodology, the experimental results, as well as findings and insights based on analyzing the results. Please list subsection
   - Don't create subsections other than the above two types (e.g. discussion for related works, experiment summary without specific evaluation results, future work).
   - The comments should include detailed information collected from input files as much as possible, but no need to well-present them in certain paper-writing form

Output the LaTeX structure with detailed comments as specified above. Do not include any other contents."""

        return await self.gpt_client.chat(prompt=prompt)

    async def find_and_fill_results(self, content_collection: str, structure: str, iteration: int) -> str:
        """Find specific experimental results and fill them into the structure in one step."""
        
        prompt = f"""Fill in specific numerical experimental results into the LaTeX structure for experiments section.

Current structure with comments:
{structure}

Content to analyze for results:
{content_collection}

Requirements:
1. RESULT EXTRACTION:
   - Find all numerical results (accuracy, precision, recall, F1, error rates, etc.)
   - Extract parameter settings and experimental conditions
   - Identify baseline comparison values
   - Collect runtime and efficiency metrics
   - Note statistical significance where available

2. RESULT PLACEMENT:
   - Fill results into appropriate sections based on experiment descriptions in comments
   - Maintain all existing LaTeX formatting and section structure
   - Keep all analysis and insights in comments
   - Present results in tables where table structures exist
   - Include results in text for single metrics
   - Group related metrics in lists

3. CONTENT PRESERVATION:
   - Keep all original section titles and hierarchy
   - Maintain all explanatory comments
   - Preserve all non-result content
   - Leave placeholders if no matching result found
   - Do not modify experiment descriptions or analysis

4. VERIFICATION:
   - Only include results clearly present in provided files
   - Verify results match experiment context
   - Do not generate or assume results
   - Maintain result precision as given in source

Output the complete LaTeX structure with actual results filled in, keeping all other content unchanged. Do not include any other contents."""

        # Get updated structure with results filled in
        updated_structure = await self.gpt_client.chat(prompt=prompt)
        
        return updated_structure

    async def detailize_subsection(self, structure: str, current_text: str, content: str, subsection: str) -> str:
        writing_template = self.get_random_template()
        
        prompt = f"""Write or revise the following subsection of the experiments section:
\subsection{{{subsection}}}

CURRENT TEXT (if any):
{current_text}

STRUCTURE INFORMATION:
{structure}

NEW CONTENT TO INCORPORATE:
{content}

REFERENCE WRITING TEMPLATE:
{writing_template}

Requirements for experiments writing:
1. CONTENT ORGANIZATION:
   - Present information in cohesive paragraphs
   - Use tables for related metrics/parameters
   - Minimize standalone equations
   - Remove placeholder information
   - Each point should be substantial and detailed

2. SUBSECTION SPECIFIC REQUIREMENTS:
   Experimental Setup:
   - Comprehensive parameter settings in tables
   - Detailed training protocols in paragraphs
   - Hardware/software specifications as needed
   - Clear evaluation metrics description

   Datasets:
   - Dataset statistics in tables
   - Preprocessing steps in paragraphs
   - Clear data split information
   - Representative examples if relevant

   Results:
   - Performance comparisons in tables
   - Analysis in detailed paragraphs
   - Statistical significance tests
   - Clear trend descriptions

   Analysis/Ablation:
   - Component analysis in paragraphs
   - Parameter studies in tables/figures
   - Detailed case studies
   - Error analysis and insights

3. WRITING STYLE:
   - Technical precision
   - Objective presentation
   - Clear justifications
   - Logical flow

Output the detailed LaTeX text for this subsection only. Do not include any other content"""

        return await self.gpt_client.chat(prompt=prompt)

    async def final_writing_checklist(self, experiments_text: str) -> str:
        prompt = f"""Review and revise the experiments section following these academic writing guidelines:

Current experiments text:
{experiments_text}

REVISION REQUIREMENTS:

1. CONTENT ORGANIZATION:
   - Information presented in meaningful paragraphs
   - Related points grouped in tables
   - Minimal equation usage
   - No placeholder text
   - Substantial coverage of each point

2. EXPERIMENTAL DETAILS:
   - Complete parameter specifications
   - Clear methodology descriptions
   - Comprehensive evaluation protocols
   - Detailed implementation notes

3. RESULTS PRESENTATION:
   - Well-structured performance tables
   - Statistical validity evidence
   - Clear metric definitions
   - Fair comparison frameworks

4. ANALYSIS DEPTH:
   - Thorough component analysis
   - Detailed ablation studies
   - Clear cause-effect relationships
   - Comprehensive error analysis

5. WRITING QUALITY:
   - Professional academic style
   - Technical precision
   - Logical flow
   - Strong experimental justifications

Output the revised experiments section incorporating all these improvements. Reply with LaTeX code only."""

        return await self.gpt_client.chat(prompt=prompt)

    async def compose_section(self, agent_dir: str, proj_dir: str, benchmark_path: str, target_paper: str) -> str:
        checkpoint_dir = self.get_checkpoint_path(target_paper)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get project directory
        workplace_dir = os.path.join(proj_dir, 'workplace/project/')
        # model_dir = os.path.join(workplace_dir, 'model/')
        
        # Read project structure and contents
        dir_tree, code_contents = self.read_project_structure(workplace_dir)

        project_summary = '***Directory Tree***:\n' + str(dir_tree) + '\n\n' + '***Code Contents***:\n' + str(code_contents)
        self.write_temp_log(project_summary, "project_summary")

        # Focus on experiment-related agent files
        agent_files = [
            'experiment_analysis_agent_iter_refine_1.json',
            'machine_learning_agent_iter_refine_1.json',
            'experiment_analysis_agent_iter_refine_2.json',
            'machine_learning_agent_iter_refine_2.json',
        ]

        # Step 1: Iterative structure generation
        structure = ""
        structure_checkpoint = self.load_checkpoint(target_paper, "structure")
        
        if structure_checkpoint:
            structure = structure_checkpoint["final_structure"]
            logging.info("Loaded structure from checkpoint")
            exit()
        else:
            for iteration in range(self.structure_iterations + 1):
                if iteration < self.structure_iterations:
                    temp_func = self.generate_or_revise_structure
                else:
                    temp_func = self.find_and_fill_results
                logging.info(f"Structure iteration {iteration + 1}/{self.structure_iterations + 1}")
                
                for idx, agent_file in enumerate(tqdm(agent_files, desc="Processing agent files")):
                    with open(os.path.join(agent_dir, agent_file), 'r') as f:
                        content = json.load(f)
                    structure = await temp_func(json.dumps(content, indent=2), structure, iteration + 1)
                
                structure = await temp_func(
                    project_summary, structure, iteration + 1)
                
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
                experiments_part = ''
                
                # First process agent contents
                for i, agent_file in enumerate(agent_files):
                    with open(os.path.join(agent_dir, agent_file), 'r') as f:
                        content = json.load(f)
                    experiments_part = await self.detailize_subsection(
                        structure, experiments_part, json.dumps(content, indent=2), subsection)
                    
                    self.write_temp_log(
                        experiments_part,
                        f"subsection_{subsection_id}_agent_{i}"
                    )
                experiments_part = await self.detailize_subsection(structure, experiments_part, project_summary, subsection)
                subsection_contents[subsection] = experiments_part
                
            self.save_checkpoint(target_paper, "subsections", subsection_contents)

        # Step 3: Fuse all subsections
        self.write_temp_log(
            json.dumps(subsection_contents, indent=2),
            "pre_fusion_subsections"
        )
        
        fused_experiments = await self.fuse_subsections(structure, subsection_contents)
        self.write_temp_log(fused_experiments, "post_fusion_experiments")

        # Step 4: Final writing checklist
        final_experiments = await self.final_writing_checklist(fused_experiments)
        self.write_temp_log(final_experiments, "post_checklist_experiments")

        # Save final output
        output_dir = f"{self.research_field}/target_sections/{self.normalize_title(target_paper)}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "experiments.tex")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_experiments)
        logging.info(f"Saved final experiments to {output_path}")

        return final_experiments

async def experiments_composing(research_field: str, instance_id: str):
    setup_logging(research_field)
    
    composer = ExperimentsComposer(research_field=research_field, structure_iterations=1)#, gpt_model='o1-mini-2024-09-12')
    
    # proj_dir = f'/data2/tjb_share/{research_field}/{instance_id}/'
    proj_dir = f'./paper_agent/{research_field}/{instance_id}/'
    # target_paper = 'Heterogeneous Graph Contrastive Learning for Recommendation'
    cache_dirs = [d for d in os.listdir(proj_dir) if d.startswith('cache_')]
    if not cache_dirs:
        raise ValueError("No cache directory found")
    agent_dir = os.path.join(proj_dir, cache_dirs[-1], 'agents')
    
    # model_dir = os.path.join(proj_dir, 'workplace/project/model/')
    # benchmark_path = f'/data2/tjb/Inno-agent/benchmark/final/{research_field}/{instance_id}.json'
    benchmark_path = f'./benchmark/final/{research_field}/{instance_id}.json'
    
    try:
        experiments = await composer.compose_section(
            agent_dir, proj_dir, benchmark_path, instance_id)
        logging.info("Experiments composition completed")
    except Exception as e:
        logging.error(f"Error during experiments composition: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(experiments_composing())