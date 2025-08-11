import os
import numpy as np
import cv2
from pdf2image import convert_from_path
from tqdm import tqdm
import subprocess

def pdf_to_twitter_video(instance_id, fps=30, dpi=150, max_width=1280):
    """
    将PDF文件转换为适合Twitter的视频
    
    参数:
        instance_id: 实例ID，用于定位PDF文件和命名输出文件
        fps: 每秒帧数，Twitter推荐30fps
        dpi: 图像分辨率
        max_width: 最大宽度，Twitter推荐1280px
    """
    # 处理文件路径
    pdf_path = f'examples/{instance_id}/paper.pdf'
    output_dir = f'examples/{instance_id}'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出路径
    temp_video_path = os.path.join(output_dir, 'paper_temp.mp4')
    final_video_path = os.path.join(output_dir, 'paper_twitter.mp4')
    
    print(f"正在将PDF转换为图像: {pdf_path}")
    # 将PDF转换为图像
    images = convert_from_path(pdf_path, dpi=dpi)
    
    print(f"共 {len(images)} 页")
    
    if not images:
        print("未能从PDF提取任何页面")
        return None
    
    # 调整图像大小以符合Twitter要求
    processed_images = []
    for img in images:
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 如果宽度超过最大宽度，按比例缩小
        if w > max_width:
            new_h = int(h * (max_width / w))
            img_array = cv2.resize(img_array, (max_width, new_h))
        
        processed_images.append(img_array)
    
    # 获取调整后的尺寸
    height, width = processed_images[0].shape[:2]
    
    # 使用H.264编码器，这是Twitter推荐的编码器
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    
    # 创建视频写入器
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # 在第一页停留较长时间
    first_frame = cv2.cvtColor(processed_images[0], cv2.COLOR_RGB2BGR)
    for _ in range(fps * 2):  # 停留2秒
        video_writer.write(first_frame)
    
    # 添加所有页面，每页停留较长时间
    for img_array in processed_images:
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # 每页停留3秒
        for _ in range(fps * 1):
            video_writer.write(frame)
    
    # 最后一页停留较长时间
    last_frame = cv2.cvtColor(processed_images[-1], cv2.COLOR_RGB2BGR)
    for _ in range(fps * 2):  # 停留2秒
        video_writer.write(last_frame)
    
    # 释放资源
    video_writer.release()
    
    # 使用FFmpeg进一步优化视频以符合Twitter要求
    try:
        # 检查是否安装了FFmpeg
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # 使用FFmpeg优化视频
        cmd = [
            'ffmpeg', '-i', temp_video_path, 
            '-c:v', 'libx264', '-profile:v', 'high', '-level:v', '4.0',
            '-pix_fmt', 'yuv420p', '-b:v', '5M', '-maxrate', '5M', '-bufsize', '10M',
            '-r', str(fps), '-movflags', '+faststart',
            '-y', final_video_path
        ]
        
        subprocess.run(cmd, check=True)
        print(f"已使用FFmpeg优化视频")
        
        # 删除临时文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        video_path = final_video_path
    except (subprocess.SubprocessError, FileNotFoundError):
        print("FFmpeg未安装或运行失败，使用原始视频")
        video_path = temp_video_path
    
    print(f"视频已创建: {video_path}")
    print(f"文件大小: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    return video_path

if __name__ == "__main__":
    # 示例用法
    instance_list = ["rotation_vq", "con_flowmatching", "dccf", "hgcl"]
    
    for instance_id in tqdm(instance_list):
        try:
            pdf_path = f'examples/{instance_id}/paper.pdf'
            if os.path.exists(pdf_path):
                pdf_to_twitter_video(
                    instance_id=instance_id,
                    fps=30,  # Twitter推荐30fps
                    dpi=300,  # 分辨率
                    max_width=1280  # Twitter推荐宽度
                )
            else:
                print(f"跳过 {instance_id}：PDF文件不存在")
        except Exception as e:
            print(f"处理 {instance_id} 时出错: {str(e)}") 