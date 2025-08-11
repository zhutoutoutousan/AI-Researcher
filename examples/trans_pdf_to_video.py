import os
import numpy as np
import cv2
from pdf2image import convert_from_path
from tqdm import tqdm

def pdf_to_video(instance_id, fps=1, quality=95, dpi=200, format='mp4'):
    """
    将PDF文件转换为视频
    
    参数:
        instance_id: 实例ID，用于定位PDF文件和命名输出文件
        fps: 每秒帧数，控制翻页速度
        quality: 视频质量 (0-100)，越高质量越好但文件越大
        dpi: 图像分辨率，越高越清晰但处理越慢
        format: 输出视频格式，可选 'mp4', 'avi'
    """
    # 处理文件路径
    pdf_path = f'examples/{instance_id}/paper.pdf'
    output_dir = f'examples/{instance_id}'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出视频路径
    video_path = os.path.join(output_dir, f'paper.{format}')
    
    print(f"正在将PDF转换为图像: {pdf_path}")
    # 将PDF转换为图像
    images = convert_from_path(pdf_path, dpi=dpi)
    
    print(f"共 {len(images)} 页")
    
    if not images:
        print("未能从PDF提取任何页面")
        return None
    
    # 获取第一张图片的尺寸作为视频尺寸
    height, width = np.array(images[0]).shape[:2]
    
    # 设置视频编码器和参数
    if format.lower() == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI格式
    
    # 创建视频写入器
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # 准备帧
    frames = []
    
    # 在第一页和最后一页停留更长时间
    first_frame = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    last_frame = cv2.cvtColor(np.array(images[-1]), cv2.COLOR_RGB2BGR)
    
    # 第一页停留2秒
    for _ in range(fps * 2):
        video_writer.write(first_frame)
    
    # 添加所有页面
    for img in images:
        # 转换PIL图像为OpenCV格式 (RGB到BGR)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    
    # 最后一页停留2秒
    for _ in range(fps * 1):
        video_writer.write(last_frame)
    
    # 释放资源
    video_writer.release()
    
    print(f"视频已创建: {video_path}")
    print(f"文件大小: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    return video_path

if __name__ == "__main__":
    # 示例用法
    instance_list = ["rotation_vq", "con_flowmatching", "dccf", "hgcl"]
    
    # 可以取消注释下面这行来只处理一个实例
    # instance_list = ["rotation_vq"]
    
    for instance_id in tqdm(instance_list):
        try:
            pdf_path = f'examples/{instance_id}/paper.pdf'
            if os.path.exists(pdf_path):
                pdf_to_video(
                    instance_id=instance_id,
                    fps=1,  # 每秒翻页数
                    quality=75,  # 视频质量
                    dpi=150,  # 分辨率
                    format='mp4'  # 输出格式，可选 'mp4' 或 'avi'
                )
            else:
                print(f"跳过 {instance_id}：PDF文件不存在")
        except Exception as e:
            print(f"处理 {instance_id} 时出错: {str(e)}")