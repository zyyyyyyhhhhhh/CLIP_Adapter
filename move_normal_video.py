import os
import shutil

# 定义文件路径
meta_file_path = '../list/train_meta.txt'
normal_videos_src_dir = '../../../dataset/UCF-Crime-Mine/UCF-Crime/all_rgbs/Normal_Videos_event'
training_videos_dest_dir = '../../../dataset/UCF-Crime-Mine/UCF-Crime/all_rgbs/Training_Normal_Videos_Anomaly'

# 读取 meta 文件
with open(meta_file_path, 'r') as meta_file:
    lines = meta_file.readlines()

# 处理每一行
for line in lines:
    line = line.strip()
    dest_path, label = line.split(',')

    # 确认标签为 Normal
    if label == 'Normal':
        # 获取源文件路径
        file_name = os.path.basename(dest_path)
        src_path = os.path.join(normal_videos_src_dir, file_name)

        # 确保目标目录存在
        if not os.path.exists(training_videos_dest_dir):
            os.makedirs(training_videos_dest_dir)

        # 目标文件路径
        dest_full_path = os.path.join(training_videos_dest_dir, file_name)

        # 移动文件
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_full_path)
            print(f"Copy: {src_path} to {dest_full_path}")
        else:
            print(f"File not found: {src_path}")
