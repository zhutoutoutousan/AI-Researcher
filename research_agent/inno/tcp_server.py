import socket
import subprocess
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--workplace", type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    assert args.workplace is not None, "Workplace is not specified"
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 12345))
    server.listen(1)

    print("Listening on port 12345...")
    def receive_all(conn, buffer_size=4096):
        data = b""
        while True:
            part = conn.recv(buffer_size)
            data += part
            if len(part) < buffer_size:
                # 如果接收的数据小于缓冲区大小，可能已经接收完毕
                break
        return data.decode()

    while True:
        conn, addr = server.accept()
        print(f"Connection from {addr}")
        while True:
            # command = conn.recv(1024).decode()
            command = receive_all(conn)
            if not command:
                break
            
            # Execute the command
            try:
                modified_command = f"/bin/bash -c 'source /home/user/micromamba/etc/profile.d/conda.sh && conda activate autogpt && cd /{args.workplace} && {command}'"
                process = subprocess.Popen(modified_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                output = ''
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    output += line
                    print(line, end='')

                exit_code = process.wait()
            except Exception as e:
                exit_code = -1
                output = f"Error running command: {str(e)}"

            # Create a JSON response
            response = {
                "status": exit_code,
                "result": output
            }
            
            # Send the JSON response
            conn.send(json.dumps(response).encode())
        conn.close()