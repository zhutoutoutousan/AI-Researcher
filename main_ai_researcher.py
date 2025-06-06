import numpy as np
import argparse
import os
import asyncio
import global_state
from dotenv import load_dotenv




def init_ai_researcher():
    a = 1

def get_args_research(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, default="benchmark/gnn.json")
    parser.add_argument('--container_name', type=str, default='paper_eval')
    parser.add_argument("--task_level", type=str, default="task1")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--workplace_name", type=str, default="workplace")
    parser.add_argument("--cache_path", type=str, default="cache")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--max_iter_times", type=int, default=0)
    parser.add_argument("--category", type=str, default="recommendation")
    args = parser.parse_args()
    return args

def get_args_paper():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research_field", type=str, default="vq")
    parser.add_argument("--instance_id", type=str, default="rotation_vq")
    args = parser.parse_args()
    return args

def main_ai_researcher(input, reference, mode):
    # if main_autoagent.mode is None:
    #     main_autoagent.mode = mode
        
    # if main_autoagent.mode != mode:
    #     model = COMPLETION_MODEL
    #     main_autoagent.mode = mode
    #     global_state.INIT_FLAG = False
    load_dotenv()
    category = os.getenv("CATEGORY")
    instance_id = os.getenv("INSTANCE_ID")
    task_level = os.getenv("TASK_LEVEL")
    container_name = os.getenv("CONTAINER_NAME")
    workplace_name = os.getenv("WORKPLACE_NAME")
    cache_path = os.getenv("CACHE_PATH")
    port = int(os.getenv("PORT"))
    max_iter_times = int(os.getenv("MAX_ITER_TIMES"))

    
    match mode:
        case 'Detailed Idea Description':
            # global INIT_FLAG
            if global_state.INIT_FLAG is False:
                global_state.INIT_FLAG = True
                current_file_path = os.path.realpath(__file__)
                current_dir = os.path.dirname(current_file_path)
                sub_dir = os.path.join(current_dir, "research_agent")
                os.chdir(sub_dir)

                from research_agent.constant import COMPLETION_MODEL
                from research_agent import run_infer_idea, run_infer_plan

                args = get_args_research()
                # category="vq"
                # instance_id="rotation_vq"
                args.instance_path = f"../benchmark/final/{category}/{instance_id}.json"
                args.task_level = task_level
                args.model = COMPLETION_MODEL
                args.container_name = container_name
                args.workplace_name = workplace_name
                args.cache_path = cache_path
                args.port = port
                args.max_iter_times = max_iter_times
                args.category = category

                run_infer_plan.main(args, input, reference)
                global_state.INIT_FLAG = False
        case 'Reference-Based Ideation':
            # clear_screen()
            if global_state.INIT_FLAG is False:
                global_state.INIT_FLAG = True
                current_file_path = os.path.realpath(__file__)
                current_dir = os.path.dirname(current_file_path)
                sub_dir = os.path.join(current_dir, "research_agent")
                os.chdir(sub_dir)

                from research_agent.constant import COMPLETION_MODEL
                from research_agent import run_infer_idea, run_infer_plan
                from research_agent.constant import COMPLETION_MODEL
                args = get_args_research()
                # category="vq"
                # instance_id="one_layer_vq"
                # args.instance_path = f"../benchmark/final/{category}/{instance_id}.json"
                # args.container_name = "paper_eval"
                # args.task_level = "task1"
                # args.model = COMPLETION_MODEL
                # args.workplace_name = "workplace"
                # args.cache_path = "cache"
                # args.port = 12356
                # args.max_iter_times = 0


                args.instance_path = f"../benchmark/final/{category}/{instance_id}.json"
                args.container_name = container_name
                args.task_level = task_level
                args.model = COMPLETION_MODEL
                args.workplace_name = workplace_name
                args.cache_path = cache_path
                args.port = port
                args.max_iter_times = max_iter_times
                args.category = category

                run_infer_idea.main(args, reference)
                global_state.INIT_FLAG = False
        case 'Paper Generation Agent':
            # clear_screen()
            if global_state.INIT_FLAG is False:
                global_state.INIT_FLAG = True

                from paper_agent import writing
                args = get_args_paper()

                research_field=category
                # instance_id="rotated_vq"
                args.research_field = research_field
                args.instance_id = instance_id

                asyncio.run(writing.writing(args.research_field, args.instance_id))
                global_state.INIT_FLAG = False
