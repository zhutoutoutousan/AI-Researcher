import yaml
import hashlib
import zipfile
import os
import json
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def read_yaml_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = yaml.safe_load(file)
    return content

def get_file_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # read file block
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def compress_folder(source_folder, destination_folder, archive_name):
    os.makedirs(destination_folder, exist_ok=True)
    
    archive_path = os.path.join(destination_folder, archive_name)
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

        for root, _, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_folder)
                zipf.write(file_path, arcname)
    
    print(f"Folder '{source_folder}' has been compressed to '{archive_path}'")

def get_md5_hash_bytext(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = json.load(file)
    return content