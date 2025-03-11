from research_agent.inno.environment.docker_env import DockerEnv

import tiktoken
from datetime import datetime

def truncate_by_tokens(env: DockerEnv, text, max_tokens = 4096, model="gpt-4o-2024-08-06"):
    from research_agent.inno.tools.terminal_tools import create_file
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = f"{env.docker_workplace}/console_output/truncated_output_{timestamp}.txt"
    create_msg = create_file(output_path, content = text, env=env)
    # 截断tokens并解码回字符串
    truncated_tokens_bos = tokens[:max_tokens//2]
    truncated_tokens_eos = tokens[-(max_tokens - len(truncated_tokens_bos)):]
    if create_msg.startswith("Error"):
        return encoding.decode(truncated_tokens_bos) + "\n...\n" + encoding.decode(truncated_tokens_eos) + "\n\nThe full console output is too long, so I want to save them into the file: {0}\n\nBut I got an error: {1}".format(output_path, create_msg)
    
    return encoding.decode(truncated_tokens_bos) + "\n...\n" + encoding.decode(truncated_tokens_eos) + "\n\nThe full console output is too long, so it is saved in the file: {0}\n\nYou may use the `File Surfer Agent` to view the full output.".format(output_path)

