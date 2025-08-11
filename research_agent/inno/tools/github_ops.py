# from metachain.util import run_command_in_container
from research_agent.inno.environment.docker_env import DockerEnv
from research_agent.constant import GITHUB_AI_TOKEN
from research_agent.inno.tools.github_client import GitHubClient
import json
from research_agent.inno.registry import register_tool

@register_tool("get_current_branch")
def get_current_branch(env: DockerEnv):
    f"""
    Get the current branch of the 'metachain'.
    """
    branch_command = f"cd {env.docker_workplace}/metachain && git branch --show-current"
    result = env.run_command(branch_command)
    if result['status'] == 0:
        return result['result'].strip()
    else:
        return f"Failed to get the current branch. Error: {result['result'].strip()}"

@register_tool("get_diff")
def get_diff(env: DockerEnv): 
    f"""
    Get the diff of the 'metachain'.
    """
    diff_command = f"cd {env.docker_workplace}/metachain && git add -N . && git diff"
    result = env.run_command(diff_command)
    if result['status'] == 0:
        return result['result'].strip()
    else:
        return f"Failed to get the diff. Error: {result['result'].strip()}"

@register_tool("stage_files")
def stage_files(env: DockerEnv, file_paths=None):
    """
    Stage the specified file changes
    
    Args:
        file_paths (list): The file paths to stage, if None, add all changes to the staging area
        
    Returns:
        dict: The operation result 
    """
    if file_paths is None:
        # add all changes to the staging area
        command = f"cd {env.docker_workplace}/metachain && git add ."
    else:
        # add specified files to the staging area
        files = ' '.join(file_paths)
        command = f"cd {env.docker_workplace}/metachain && git add {files}"
    
    result = env.run_command(command)
    return result

@register_tool("push_changes")
def push_changes(env: DockerEnv, commit_message, file_paths=None):
    """
    Push the selected changes to the remote repository
    
    Args:
        commit_message (str): The commit message
        file_paths (list): The file paths to commit, if None, commit all changes
        
    Returns:
        dict: The push result
    """
    # stage the files
    # if file_paths:
    stage_result = stage_files(env, file_paths)
    if stage_result['status'] != 0:
        return json.dumps({'status': 'error', 'message': f"Failed to stage files: {stage_result['result']}"}, indent=4)
    
    commands = [
        f"cd {env.docker_workplace}/metachain",
        f'git commit -m "{commit_message}"',
        "git push origin $(git branch --show-current)"
    ]
    
    command = " && ".join(commands)
    result = env.run_command(command)
    
    if result['status'] == 0:
        return f"push success. {result['result']}"
    else:
        return f"push failed. {result['result']}"

@register_tool("submit_pull_request")
def submit_pull_request(title: str, body: str, target_branch: str): 
    """
    Submit a Pull Request
    
    Args:
        title: PR title
        body: PR description
        target_branch: target branch
    """
    # initialize GitHub client
    github = GitHubClient(GITHUB_AI_TOKEN)
    
    # check authentication
    auth_result = github.check_auth()
    if auth_result['status'] != 0:
        return auth_result
    
    # create a pull request
    pr_result = github.create_pull_request(
        repo="tjb-tech/metachain",
        title=title,
        body=body,
        head=get_current_branch(),
        base=target_branch
    )
    if pr_result['status'] == 0:
        return f"PR created successfully: {json.dumps(pr_result, indent=4)}"
    else:
        return f"PR creation failed: {json.dumps(pr_result, indent=4)}"

# def create_pull_request(title, body, target_branch):
#     """
#     Create a Pull Request to the target branch
    
#     Args:
#         title (str): The title of the PR
#         body (str): The description content of the PR
#         target_branch (str): The target branch name
    
#     Returns:
#         dict: PR creation result
#     """
    
#     # use gh to create a PR. make sure the gh cli is installed in the container and the github token is set
#     pr_command = f"""cd /{DOCKER_WORKPLACE_NAME}/metachain && \
#         gh pr create \
#         --title "{title}" \
#         --body "{body}" \
#         --base {target_branch} \
#         --head $(git branch --show-current)"""
    
#     result = run_command_in_container(pr_command)
    
#     if result['status'] == 0:
#         return f"PR created successfully: {result['result']}"
#     else:
#         return f"PR creation failed: {result['result']}"
if __name__ == "__main__":
    from rich import print
    print("Current branch: " + get_current_branch())
    print("Diff: " + get_diff())
    print(push_changes(commit_message="test"))
    print(submit_pull_request(title="test", body="test", target_branch="test_pull_1107"))