#!/bin/bash

cd path/to/AI-Researcher/paper_agent

export OPENAI_API_KEY=sk-xxx


research_field=vq
instance_id=rotated_vq

python path/to/AI-Researcher/paper_agent/writing.py --research_field ${research_field} --instance_id ${instance_id}
