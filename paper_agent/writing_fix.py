import os

def clean_tex_file(filepath):
    """
    清理 .tex 文件开头和结尾包含 ``` 的行
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 去除首行包含 ``` 的行
    while lines and '```' in lines[0]:
        # print(f"[{os.path.basename(filepath)}] 删除首行: {lines[0].strip()}")
        lines.pop(0)

    # 去除末行包含 ``` 的行
    while lines and '```' in lines[-1]:
        # print(f"[{os.path.basename(filepath)}] 删除尾行: {lines[-1].strip()}")
        lines.pop(-1)

    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def clean_tex_files_in_folder(folder_path):
    """
    遍历文件夹，处理所有 .tex 文件
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.tex'):
                full_path = os.path.join(root, file)
                clean_tex_file(full_path)


def process_tex_file(tex_path, bib_output_path):
    with open(tex_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    bib_started = False
    bib_lines = []

    for i, line in enumerate(lines):
        # 删除以 \bib 开头的行
        if line.strip().startswith('\\bib'):
            # print(f"删除行（\\bib 开头）: {line.strip()}")
            continue

        # 查找第一个以 @ 开头的行，标志 bib 开始
        if not bib_started and line.strip().startswith('@'):
            bib_started = True
            # print(f"检测到 bib 条目开始于第 {i+1} 行")
            bib_lines.append(line)
        elif bib_started:
            bib_lines.append(line)
        else:
            new_lines.append(line)

    # 写回清理后的 .tex 文件
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
        # print(f"更新后的 tex 文件已写入：{tex_path}")

    # 如果提取到了 bib 内容，写入 .bib 文件
    if bib_lines:
        with open(bib_output_path, 'w', encoding='utf-8') as f:
            f.writelines(bib_lines)
            # print(f"BibTeX 条目已写入：{bib_output_path}")
    else:
        # print("未找到任何以 @ 开头的 BibTeX 条目，未创建 .bib 文件。")
        print("all files are right")


# # === 使用方式 ===
# # 请将此路径替换为你的实际文件夹路径
# target_folder = "./paper_agent/final_paper"
# clean_tex_files_in_folder(target_folder)

# tex_file_path = './paper_agent/final_paper/related_work.tex'  # 替换为你的 .tex 文件路径
# bib_file_path = './paper_agent/final_paper/iclr2025_conference.bib'  # 目标 .bib 文件名
# process_tex_file(tex_file_path, bib_file_path)