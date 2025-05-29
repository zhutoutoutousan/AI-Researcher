import os
import subprocess



def compile_latex_project(project_dir, main_tex_file):
    """
    编译包含多个文件的LaTeX项目
    
    Args:
        project_dir (str): 项目目录的路径
        main_tex_file (str): 主tex文件的名称
    """
    try:
        # 切换到项目目录
        original_dir = os.getcwd()
        os.chdir(project_dir)
        
        # 获取主文件的完整路径
        main_file_path = os.path.join(project_dir, main_tex_file)
        
        print(f"开始编译: {main_file_path}")
        
        # 运行编译命令两次(处理目录和引用)
        for i in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', main_tex_file],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"编译错误:\n{result.stderr}")
                print(f"编译错误:\n{result.stdout}")
                return False
        
        # 检查PDF是否生成
        pdf_file = os.path.splitext(main_tex_file)[0] + '.pdf'
        if os.path.exists(pdf_file):
            print(f"PDF生成成功: {os.path.join(project_dir, pdf_file)}")
            return True
        else:
            print("PDF生成失败")
            return False
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False
        
    finally:
        # 恢复原始工作目录
        os.chdir(original_dir)

# 使用示例
# project_directory = "/data2/tjb/writing_part/paper_agent/tex_project"
# main_file = "main.tex"
project_directory = "./paper_agent/final_paper"
main_file = "iclr2025_conference.tex"
compile_latex_project(project_directory, main_file)