import os
import json
import asyncio
import logging
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.openai_utils import GPTClient
from section_composer import SectionComposer, setup_logging

class AbstractComposer(SectionComposer):
    def __init__(self, research_field: str, structure_iterations: int = 2):
        super().__init__(research_field, "abstract", structure_iterations)

    def read_section_content(self, target_paper: str, section_name: str) -> str:
        """Read content from an existing section file"""
        normalized_title = self.normalize_title(target_paper)
        section_path = f"{self.research_field}/target_sections/{normalized_title}/{section_name}.tex"
        try:
            with open(section_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.warning(f"Section file {section_path} not found")
            return ""

    async def generate_or_revise_structure(self, content, current_structure, iteration):
        prompt = f"""Based on the given content, generate or revise the abstract structure, using latex format.
Current iteration: {iteration}/{self.structure_iterations}

Current structure (if exists):
{current_structure}

Content to analyze:
{content}

Guidelines for structure generation:
1. STRUCTURE FORMAT:
   \begin{{abstract}}
   
   % [Problem Context]
   % - Research area and problem statement
   % - Current challenges or limitations
   
   % [Proposed Solution]
   % - Main approach and key innovations
   % - Technical highlights
   
   % [Results and Impact]
   % - Key experimental findings
   % - Practical significance
   
   \end{{abstract}}

2. CONTENT REQUIREMENTS:
   - Keep structure focused and concise
   - Ensure logical flow from problem to results
   - Highlight key contributions
   - Emphasize practical impact

Output only the LaTeX structure with comments as specified above."""

        return await self.gpt_client.chat(prompt=prompt)

    async def detailize_subsection(self, structure, current_text, content):
        writing_template = self.get_random_template()
        prompt = f"""Write a comprehensive abstract based on the provided structure and content.

CURRENT ABSTRACT VERSION (if any):
{current_text}

STRUCTURE:
{structure}

PAPER CONTENT TO SUMMARIZE:
{content}

Writing Guidelines:

1. PROBLEM CONTEXT:
   - State research area and problem clearly
   - Shortly recap existing research line
   - Research motivation
   - Present specific problem
   - Highlight key challenges

2. PROPOSED SOLUTION:
   - Present main approach
   - Highlight technical innovations and its relation to motivation

3. RESULTS:
   - State primary findings

STYLE REQUIREMENTS:
- Use precise, clear language
- Maintain formal academic tone
- Avoid technical jargon
- Focus on key points only
- Keep to 150-250 words
- Write as one cohesive paragraph
- No citations or references
- No future work mentions
- Use latex, don't use markdown

Output the complete LaTeX text for the abstract section. Do not output any other content."""

        return await self.gpt_client.chat(prompt=prompt)

    async def final_writing_checklist(self, abstract_text: str) -> str:
        prompt = f"""Review and revise the abstract section following these guidelines:

Current abstract text:
{abstract_text}

CHECKLIST FOR REVISION:

1. CONTENT COMPLETENESS:
   - Clear problem statement
   - Main methodology explained
   - Key results included
   - Significance highlighted

2. WRITING QUALITY:
   - No redundant information
   - Clear logical flow
   - Concise presentation
   - Appropriate length

3. TECHNICAL ACCURACY:
   - Precise terminology
   - Correct technical claims
   - Clear methodology description
   - Accurate result reporting

4. STYLE AND FORMAT:
   - Professional academic tone
   - No jargon or undefined terms
   - Single paragraph structure
   - No citations or references
   - No future work mentions

Output the revised abstract section incorporating all these improvements. Reply with LaTeX text only."""

        return await self.gpt_client.chat(prompt=prompt)

    async def compose_section(self, target_paper: str) -> str:
        checkpoint_dir = self.get_checkpoint_path(target_paper)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Read existing sections
        introduction = self.read_section_content(target_paper, "introduction")
        methodology = self.read_section_content(target_paper, "methodology")
        experiments = self.read_section_content(target_paper, "experiments")
        content_bundle = introduction + '\n\n' + methodology + '\n\n' + experiments

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
                    content_bundle, structure, iteration + 1)
                self.write_temp_log(structure, f"iteration_{iteration+1}_final")
            
            self.save_checkpoint(target_paper, "structure", {
                "final_structure": structure
            })

        # Step 2: Write complete abstract
        final_abstract = await self.detailize_subsection(structure, "", content_bundle)
        self.write_temp_log(final_abstract, "initial_abstract")

        
        # Save final output
        output_dir = f"{self.research_field}/target_sections/{self.normalize_title(target_paper)}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "abstract.tex")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_abstract)
        logging.info(f"Saved final abstract to {output_path}")

        return final_abstract

async def abstract_composing(research_field: str, instance_id: str):
    setup_logging(research_field)
    
    composer = AbstractComposer(research_field=research_field, structure_iterations=2)
    # target_paper = 'Heterogeneous Graph Contrastive Learning for Recommendation'
    
    try:
        abstract = await composer.compose_section(instance_id)
        logging.info("Abstract composition completed")
    except Exception as e:
        logging.error(f"Error during abstract composition: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(abstract_composing())