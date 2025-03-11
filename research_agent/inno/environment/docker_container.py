import os
import subprocess
from research_agent.constant import GITHUB_AI_TOKEN, AI_USER, BASE_IMAGES
import time
from research_agent.inno.util import run_command_in_container

def init_container(workplace_name, container_name, test_pull_name = 'test_pull_1010', task_name = 'test_task', git_clone = False, setup_package = 'setup_package'):
    # get the current working directory's subfolder path
    workplace = os.path.join(os.getcwd(), workplace_name)

    # check if the container exists
    container_check_command = ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"]
    existing_container = subprocess.run(container_check_command, capture_output=True, text=True)

    os.makedirs(workplace, exist_ok=True)
    # cp_command = ["cp", "tcp_server.py", workplace]
    if not os.path.exists(os.path.join(workplace, 'tcp_server.py')):
        unzip_command = ["tar", "-xzvf", f"packages/{setup_package}.tar.gz", "-C", workplace]
        subprocess.run(unzip_command)
    if git_clone:
        if not os.path.exists(os.path.join(workplace, 'metachain')):
            git_command = ["cd", workplace, "&&", "git", "clone", "-b", test_pull_name, f"https://{AI_USER}:{GITHUB_AI_TOKEN}@github.com/tjb-tech/metachain.git"]
            git_command = " ".join(git_command)
            
            result = subprocess.run(git_command, shell=True)
            if result.returncode != 0:
                raise Exception(f"Failed to clone the repository. Please check your internet connection and try again.")
            # create a new branch
        new_branch_name = f"{test_pull_name}_{task_name}"
        create_branch_command = f"cd {workplace}/metachain && git checkout -b {new_branch_name}"
        result = subprocess.run(create_branch_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(Exception(f"Failed to create and switch to new branch. Error: {result.stderr}"))
            switch_branch_command = f"cd {workplace}/metachain && git checkout {new_branch_name}"
            result = subprocess.run(switch_branch_command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to switch to new branch. Error: {result.stderr}")
            else:
                print(f"Successfully switched to new branch: {new_branch_name}")
        else:
            print(f"Successfully created and switched to new branch: {new_branch_name}")

    if existing_container.stdout.strip() == container_name:
        # check if the container is running
        running_check_command = ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"]
        running_container = subprocess.run(running_check_command, capture_output=True, text=True)

        if running_container.stdout.strip() == container_name:
            print(f"Container '{container_name}' is already running. Skipping creation.")
            return  # container is already running, skip creation
        else:
            # container exists but is not running, start it
            start_command = ["docker", "start", container_name]
            subprocess.run(start_command)
            print(f"Container '{container_name}' has been started.")
            return
    
    # if the container does not exist, create and start a new container
    docker_command = [
        "docker", "run", "-d", "--name", container_name, "--user", "root",
        "-v", f"{workplace}:/{workplace_name}",
        "-w", f"/{workplace_name}", "-p", "12345:12345", BASE_IMAGES,
        "/bin/bash", "-c", 
        f"python3 /{workplace_name}/tcp_server.py --workplace {workplace_name}"
    ]
    # execute the docker command
    result = subprocess.run(docker_command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to start container: {result.stderr}")
    if wait_for_container_ready(container_name, timeout=60):
        print(f"Container '{container_name}' has been created and started.")

def wait_for_container_ready(container_name, timeout=30):
    """using subprocess to check if the container is running"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and "true" in result.stdout.lower():
            # 额外检查 tcp_server 是否运行
            try:
                result = run_command_in_container('ps aux')
                if "tcp_server.py" in result['result']:
                    return True
            except Exception as e:
                pass
            
        time.sleep(1)
        
    raise TimeoutError(f"Container {container_name} failed to start within {timeout} seconds")