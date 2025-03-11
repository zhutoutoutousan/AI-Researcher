import os
import json
import asyncio
import logging
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.openai_utils import GPTClient
from section_composer import SectionComposer, setup_logging

class ConclusionComposer(SectionComposer):
    def __init__(self, research_field: str, structure_iterations: int = 2):
        super().__init__(research_field, "conclusion", structure_iterations)

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
        prompt = f"""Based on the given content, generate or revise the conclusion structure, using latex format.
Current iteration: {iteration}/{self.structure_iterations}

Current structure (if exists):
{current_structure}

Content to analyze:
{content}

Guidelines for structure generation:
1. STRUCTURE FORMAT:
   \section{{Conclusion}}
   
   % [Summary of Work]
   % - Brief recap of problem and motivation
   % - Key technical innovations
   % - Main experimental findings
   
   % [Future Work]
   % - Potential improvements
   % - New research directions
   % - Open challenges

2. CONTENT REQUIREMENTS:
   - Maintain high-level perspective
   - Focus on significance and implications
   - Balance summary with forward-looking insights
   - Connect specific findings to broader impact

Output only the LaTeX structure with comments as specified above."""

        return await self.gpt_client.chat(prompt=prompt)

    async def detailize_subsection(self, structure, current_text, content):
        writing_template = self.get_random_template()
        prompt = f"""Write a comprehensive conclusion section based on the provided structure and content.

CURRENT CONCLUSION VERSION (if any):
{current_text}

STRUCTURE:
{structure}

PAPER CONTENT TO SUMMARIZE:
{content}

Writing Guidelines:

1. SUMMARY OF WORK:
   - Begin with concise problem statement
   - Highlight main technical contributions
   - Summarize key experimental results
   - Keep technical details minimal

2. FUTURE WORK (only 1 sentence):
   - Identify meaningful research directions
   - Suggest potential improvements
   - Discuss remaining challenges
   - Keep suggestions concrete and justified
   - Connect to current limitations

STYLE REQUIREMENTS:
- Use concise, clear language
- Maintain formal academic tone
- Avoid introducing new concepts
- Focus on high-level insights
- Balance critique with optimism
- Use latex, and do not use markdown syntax
- **Keep length to only 1 short paragraph**

Output the complete LaTeX text for the conclusion section. Do not output any other content."""

        return await self.gpt_client.chat(prompt=prompt)

    async def final_writing_checklist(self, conclusion_text: str) -> str:
        prompt = f"""Review and revise the conclusion section following these guidelines:

Current conclusion text:
{conclusion_text}

CHECKLIST FOR REVISION:

1. CONTENT COMPLETENESS:
   - Verify all key contributions are mentioned
   - Check that important results are summarized
   - Ensure future work is meaningful
   - Confirm broader impact is discussed

2. WRITING QUALITY:
   - Remove redundant statements
   - Strengthen logical flow
   - Improve paragraph transitions
   - Enhance clarity and conciseness

3. TONE AND STYLE:
   - Maintain balanced perspective
   - Avoid overly strong claims
   - Keep future work realistic
   - Use appropriate academic language

4. STRUCTURAL ELEMENTS:
   - Verify proper paragraph organization
   - Check LaTeX formatting
   - Ensure appropriate length (1 short paragraph)
   - Validate section coherence

Output the revised conclusion section incorporating all these improvements. Reply with LaTeX text only."""

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

        # Step 2: Write complete conclusion
        final_conclusion = await self.detailize_subsection(structure, "", content_bundle)
        self.write_temp_log(final_conclusion, "final_conclusion")

        # Save final output
        output_dir = f"{self.research_field}/target_sections/{self.normalize_title(target_paper)}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "conclusion.tex")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_conclusion)
        logging.info(f"Saved final conclusion to {output_path}")

        return final_conclusion

async def conclusion_composing(research_field: str, instance_id: str):
    setup_logging(research_field)
    
    composer = ConclusionComposer(research_field=research_field, structure_iterations=2)
    # target_paper = 'Heterogeneous Graph Contrastive Learning for Recommendation'
    
    try:
        conclusion = await composer.compose_section(instance_id)
        logging.info("Conclusion composition completed")
    except Exception as e:
        logging.error(f"Error during conclusion composition: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(conclusion_composing())