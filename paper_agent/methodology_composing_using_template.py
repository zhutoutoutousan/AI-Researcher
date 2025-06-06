import os
import json
import asyncio
import logging
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from benchmark_collection.utils.openai_utils import GPTClient
from paper_agent.section_composer import SectionComposer, setup_logging

class MethodologyComposer(SectionComposer):
    def __init__(self, research_field: str, structure_iterations: int = 3):
        super().__init__(research_field, "methodology", structure_iterations)

    def read_model_code(self, model_dir):
        """Combine all Python files in the model directory"""
        combined_code = []
        for filename in os.listdir(model_dir):
            if filename.endswith('.py'):
                with open(os.path.join(model_dir, filename), 'r') as f:
                    combined_code.append(f"# File: {filename}\n{f.read()}\n")
        return '\n'.join(combined_code)

    async def generate_or_revise_structure(self, content, current_structure, iteration):
        prompt = f"""Based on the given content, generate or revise the technical methodology structure of the proposed method, using latex format.
Current iteration: {iteration}/{self.structure_iterations}

Current structure (if exists):
{current_structure}

Content to analyze:
{content}

Guidelines for structure generation:
1. FOCUS ON TECHNICAL METHODOLOGY:
   - Include only the technical components and mechanisms of the proposed method (e.g. a machine learning model)
   - Exclude experimental settings, configurations, and evaluation procedures (which may probably occure in the content. Ignore them)

2. SECTION HIERARCHY:
   - Main section should be the name of the Proposed Method (with latex command \section{{Name_of_Proposed_Method}})
   - Use subsections for major components under the entire proposed method (e.g., encoders, architectures, learning objectives), with latex commands \subsection{{...}} and \subsubsection{{...}}
   - Use subsubsections for detailed mechanisms within major components
   - Ensure logical flow from basic components to advanced mechanisms

3. REQUIRED COMMENTS:
   Add latex comments (start with %) under the \section or \subsection or \subsubsection commands to explain the following:

   For the entire "Proposed Method" section:
   - Overview of the technical approach (what techniques are used to achieve what goal)
   - Functionalities of different components (subsections)
   - How different components (subsections) work together. The reader should get a global picture of the entire framework with this description
   
   For each subsection and subsubsection:
   - Technical purpose of this component
   - Connection to other components
   - Key technical innovations or mechanisms
   - A brief introduction to the component

   For each subsection and the entire proposed framework, give an explicit workflow chart for the specific subsection or the entire framework, using text

   For each subsection, give clear definitions on the input and output of the component, from where it get the input, and to where the output is used

4. STRUCTURE FORMAT:
   \section{{Proposed Method}}
   % [Overall method description and component relations]
   % [Input and output of the entire framework]
   % [workflow of the entire framework]
   
   \subsection{{Component 1}}
   % [Technical purpose and relations]
   % [Input and output of component 1]
   % [workflow of component 1]
   
   \subsection{{Component 2}}
   % [Technical purpose and relations]
   % [Input and output of component 2]
   % [workflow of component 2]
   
   \subsubsection{{Component 2.1}}
   % [Technical purpose and relations]

   Note that subsections are first-level modules of the proposed method. subsubsections are either 1. second-level submodules that are relatively independent and important, or 2. aspects that are important to highlight to better introduce the module.

Output only the LaTeX structure with comments as specified above. Note again that you should include only model designs using a professional writing style for academic research in AI domains, exclude any implementation details (e.g. hyperparameter configurations, coding details), experimental settings, or evaluation procedures."""

        return await self.gpt_client.chat(prompt=prompt)

    async def detailize_subsection(self, structure, current_text, content, subsection):
        # Get a random writing template
        writing_template = self.get_random_template()
        
        prompt = f"""Revise or write the following subsection of the methodology section:
\subsection{{{subsection}}}

CURRENT TEXT (if any):
{current_text}

Note: This is an iterative editing process. If current text exists:
1. Build upon and improve the existing content
2. Add missing technical details
3. Refine the writing while preserving valid technical descriptions
4. Maintain consistency with previously written parts

STRUCTURE INFORMATION:
{structure}

Note: The structure above provides high-level information about:
1. The overall architecture and components of the method
2. The purpose and role of each component
3. How components interact with each other
4. The workflow of the entire system
Use this information to understand the big picture and component relationships, NOT as writing guidelines.

NEW TECHNICAL CONTENT TO INCORPORATE:
{content}

Note: The content above contains specific technical details about:
1. Model architectures and computations
2. Mathematical formulations
3. Algorithm workflows
4. Implementation details
Use this information to write concrete technical descriptions that are missing from or can improve the current text.

REFERENCE WRITING TEMPLATE:
{writing_template}

Note: This template is for reference only. Use it to understand:
1. Common academic writing patterns (e.g., how to introduce a component, present equations, explain benefits)
2. Types of content to include (e.g., motivation, technical details, mathematical formulations)
3. Logical flow of technical presentations
4. Professional academic writing style

DO NOT:
- Follow the template word by word
- Copy its exact sentence structures
- Force your content to fit its specific format

Instead:
- Write naturally while incorporating similar elements (motivation, technical details, equations, etc.)
- Adapt the writing style to best present your specific technical content
- Maintain similar levels of technical depth and academic rigor

Requirements:
1. If current text exists:
   - Preserve valid technical content
   - Maintain consistent writing style
   - Add missing technical details
   - Improve clarity and organization
2. If starting from scratch:
   - Write comprehensive technical content
   - Follow academic writing conventions
3. In both cases:
   - Include necessary technical details from the new content
   - Ensure alignment with the structure's component descriptions
   - Use proper LaTeX formatting
   - Create smooth transitions
   - Focus on technical precision

Output the detailed LaTeX text for this subsection only."""

        return await self.gpt_client.chat(prompt=prompt)

    async def final_writing_checklist(self, methodology_text: str) -> str:
        prompt = f"""Review and revise the methodology section following these academic writing guidelines:

Current methodology text:
{methodology_text}

CHECKLIST FOR REVISION:

1. ACADEMIC WRITING STYLE:
   - Remove any markdown-style formatting
   - Remove any code-style documentation
   - Use formal academic language and terminology
   - Maintain consistent technical writing style throughout

2. MATHEMATICAL FORMULATION:
   - Verify correctness of all mathematical notations and equations
   - Ensure consistent variable naming
   - Check equation numbering and references
   - Avoid using too long plain text in equations

3. ACADEMIC WRITING WITH MATH:
   - Ensure that all important technical modules and mechanisms are described with math equations and well-defined math notations, even they have been well-described using natural languages
   - Avoid writing too simple math equations in non-inline equations. To address such cases, you may display 2 or 3 correlated simple equations together, or show more in-depth details for the mechanism using equations.

4. CONTENT FOCUS:
   - Reduce explanations of commonly known concepts
   - Use \cite{{}} for well-established methods instead of detailed explanations. If you don't know real papers to cite, you may also simplly describe what kind of references you are referring to.
   - Concentrate on novel contributions and key technical components
   - Ensure proper balance between overview and technical depth

5. SECTION TITLES:
   - Replace generic subsection titles with context-specific ones
   - Emphasize novelty and technical focus in titles
   - Reflect the specific application domain and unique aspects
   Examples:
   - Instead of "Embedding Layer" → "Context-Aware Knowledge Graph Embedding"
   - Instead of "Attention Mechanism" → "Cross-Modal Attention for Knowledge Integration"
   - Instead of "Loss Function" → "Multi-Task Knowledge Distillation Objective"
   - But remember don't make the titles too long, just 3-6 words is fine.

Output the revised methodology section incorporating all these improvements while maintaining the core technical content. Reply your latex without any additional explanations."""

        return await self.gpt_client.chat(prompt=prompt)

    async def compose_section(self, agent_dir: str, model_dir: str, benchmark_path: str, target_paper: str) -> str:
        checkpoint_dir = self.get_checkpoint_path(target_paper)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        agent_files = [
            'prepare_agent.json',
            'survey_agent.json',
            'coding_plan_agent.json',
            'machine_learning_agent.json',
            'judge_agent.json',
            'machine_learning_agent_iter_submit.json',
            'experiment_analysis_agent_iter_refine_1.json',
            'machine_learning_agent_iter_refine_1.json',
        ]
        combined_code = self.read_model_code(model_dir)

        # Step 1: Iterative structure generation
        structure = ""
        structure_checkpoint = self.load_checkpoint(target_paper, "structure")
        
        if structure_checkpoint:
            structure = structure_checkpoint["final_structure"]
            logging.info("Loaded structure from checkpoint")
        else:
            for iteration in range(self.structure_iterations):
                logging.info(f"Structure iteration {iteration + 1}/{self.structure_iterations}")
                
                structure = await self.generate_or_revise_structure(
                    combined_code, structure, iteration + 1)

                # Process agent files
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
                methodology_part = ''
                
                # Process code content
                methodology_part = await self.detailize_subsection(
                    structure, methodology_part, combined_code, subsection)
                self.write_temp_log(
                    methodology_part,
                    f"subsection_{subsection_id}_code"
                )
                
                # Process agent contents
                for i, agent_file in enumerate(agent_files):
                    with open(os.path.join(agent_dir, agent_file), 'r') as f:
                        content = json.load(f)
                    methodology_part = await self.detailize_subsection(
                        structure, methodology_part, json.dumps(content, indent=2), subsection)
                    
                    self.write_temp_log(
                        methodology_part,
                        f"subsection_{subsection_id}_agent_{i}"
                    )
                    
                    subsection_contents[subsection] = methodology_part
                    self.save_checkpoint(target_paper, "subsections", subsection_contents)

        # Step 3: Fuse all subsections
        self.write_temp_log(
            json.dumps(subsection_contents, indent=2),
            "pre_fusion_subsections"
        )
        
        fused_methodology = await self.fuse_subsections(structure, subsection_contents)
        self.write_temp_log(fused_methodology, "post_fusion_methodology")

        # Step 4: Final writing checklist
        final_methodology = await self.final_writing_checklist(fused_methodology)
        self.write_temp_log(final_methodology, "post_checklist_methodology")

        # Save final output
        output_dir = f"{self.research_field}/target_sections/{self.normalize_title(target_paper)}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "methodology.tex")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_methodology)
        logging.info(f"Saved final methodology to {output_path}")

        return final_methodology

async def methodology_composing(research_field: str, instance_id: str):
    # research_field = "vq"
    # instance_id = "rotation_vq"
    setup_logging(research_field)
    
    composer = MethodologyComposer(research_field=research_field, structure_iterations=1)
    
    # proj_dir = f'/data2/tjb_share/{research_field}/{instance_id}/'
    proj_dir = f'./paper_agent/{research_field}/{instance_id}/'
    # target_paper = "Knowledge Graph Self-Supervised Rationalization for Recommendation"
    # target_paper = 'Heterogeneous Graph Contrastive Learning for Recommendation'
    cache_dirs = [d for d in os.listdir(proj_dir) if d.startswith('cache_')]
    if not cache_dirs:
        raise ValueError("No cache directory found")
    agent_dir = os.path.join(proj_dir, cache_dirs[-1], 'agents')
    
    model_dir = os.path.join(proj_dir, 'workplace/project/model/')
    # benchmark_path = f'/data2/tjb/Inno-agent/benchmark/final/{research_field}/{instance_id}.json'
    benchmark_path = f'./benchmark/final/{research_field}/{instance_id}.json'
    # sss
    try:
        methodology = await composer.compose_section(
            agent_dir, model_dir, benchmark_path, instance_id)
        logging.info("Methodology composition completed")
    except Exception as e:
        logging.error(f"Error during methodology composition: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(methodology_composing())