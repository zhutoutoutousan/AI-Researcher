import os
import numpy as np
import imageio
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

def pdf_to_gif(instance_id, fps=1, quality=60, dpi=200):
    """
    将PDF文件转换为GIF动画
    
    参数:
        instance_id: 实例ID，用于定位PDF文件和命名输出文件
        fps: 每秒帧数，控制翻页速度
        quality: 图像质量 (0-100)，越低文件越小
        dpi: 图像分辨率，越高越清晰但文件越大
    """
    # 处理文件路径
    pdf_path = f'examples/{instance_id}/paper.pdf'
    output_dir = f'examples/{instance_id}'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 临时文件路径
    temp_gif = os.path.join(output_dir, 'temp_paper.gif')
    final_gif = os.path.join(output_dir, 'paper.gif')
    
    print(f"正在将PDF转换为图像: {pdf_path}")
    # 将PDF转换为图像
    images = convert_from_path(pdf_path, dpi=dpi)
    
    print(f"共 {len(images)} 页")
    
    # 转换为numpy数组
    frames = [np.array(img) for img in images]
    
    # 在第一页和最后一页停留更长时间
    first_frame = frames[0]
    last_frame = frames[-1]
    
    extended_frames = []
    # 第一页停留3秒
    extended_frames.extend([first_frame] * (fps * 2))
    # 添加所有页面
    extended_frames.extend(frames)
    # 最后一页停留3秒
    extended_frames.extend([last_frame] * (fps * 2))
    
    print("正在生成GIF...")
    # 保存为临时GIF
    imageio.mimsave(temp_gif, extended_frames, fps=fps)
    
    # 使用PIL进一步优化GIF
    img = Image.open(temp_gif)
    img.save(final_gif, 
            save_all=True, 
            optimize=True,  # 启用GIF优化
            quality=quality,
            loop=0)  # 无限循环
    
    # 删除临时文件
    if os.path.exists(temp_gif):
        os.remove(temp_gif)
    
    print(f"GIF已创建: {final_gif}")
    print(f"文件大小: {os.path.getsize(final_gif) / (1024*1024):.2f} MB")
    
    return final_gif

if __name__ == "__main__":
    # 示例用法
    instance_list = ["rotation_vq", "con_flowmatching", "dccf", "fsq", "gnn_difformer", "gnn_nodeformer", "hgcl"]

    # instance_list = ["rotation_vq"]
    
    for instance_id in tqdm(instance_list):
        try:
            pdf_path = f'examples/{instance_id}/paper.pdf'
            if os.path.exists(pdf_path):
                pdf_to_gif(
                    instance_id=instance_id,
                    fps=1,  # 每秒翻页数
                    quality=100,  # 图像质量
                    dpi=500  # 分辨率
                )
            else:
                print(f"跳过 {instance_id}：PDF文件不存在")
        except Exception as e:
            print(f"处理 {instance_id} 时出错: {str(e)}")