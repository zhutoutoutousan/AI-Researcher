from PIL import Image, ImageDraw
import numpy as np
import imageio
import os
from tqdm import tqdm
# 加载长图

def trans_fig(instance_id):
    long_image = Image.open(f'examples/{instance_id}/code.png')
    width, height = long_image.size

    # 设置参数
    frame_height = 2000  # 每帧显示的高度
    step = 10  # 每帧滚动的像素数
    fps = 20  # 每秒帧数

    # 图像质量优化参数
    quality = 60  # 图像质量 (0-100)，越低文件越小


    # 创建帧
    frames = []

    # 在第一帧停留2秒
    first_frame = long_image.crop((0, 0, width, min(frame_height, height)))
    for _ in range(fps * 2):  # 停留2秒
        frames.append(np.array(first_frame))

    # 继续正常滚动
    for y_offset in range(0, height - frame_height, step):
        # 裁剪当前视图
        frame = long_image.crop((0, y_offset, width, y_offset + frame_height))
        frames.append(np.array(frame))

    # 临时文件路径
    temp_gif = f'examples/{instance_id}/temp_scrolling_code.gif'
    final_gif = f'examples/{instance_id}/scrolling_code.gif'

    # 保存为临时GIF
    imageio.mimsave(temp_gif, frames, fps=fps)

    # 使用PIL进一步优化GIF
    img = Image.open(temp_gif)
    img.save(final_gif, 
            save_all=True, 
            optimize=True,  # 启用GIF优化
            quality=quality,
            loop=0)

    # 删除临时文件
    if os.path.exists(temp_gif):
        os.remove(temp_gif)

    print(f"GIF created at {final_gif}")
    print(f"File size: {os.path.getsize(final_gif) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # trans_list = ["con_flowmatching", "dccf", "fsq", "gnn_difformer", "gnn_nodeformer", "hgcl"]
    trans_list = ["gnn_nodeformer"]
    for instance_id in tqdm(trans_list):
        trans_fig(instance_id)