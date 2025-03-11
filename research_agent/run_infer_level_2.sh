
current_dir=$(dirname "$(readlink -f "$0")")
cd $current_dir
export DOCKER_WORKPLACE_NAME=workplace_paper

export BASE_IMAGES=tjbtech1/paperagent:latest

export COMPLETION_MODEL=claude-3-5-sonnet-20241022
export CHEEP_MODEL=claude-3-5-haiku-20241022

category=vq
instance_id=one_layer_vq
export GPUS='"device=0,1"'

python run_infer_idea.py --instance_path ../benchmark/final/${category}/${instance_id}.json --container_name paper_eval --model $COMPLETION_MODEL --workplace_name workplace --cache_path cache --port 12372 --max_iter_times 0 --category ${category}

