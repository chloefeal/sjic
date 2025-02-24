def convert_mp4_to_jpg(video_path, output_folder):
  import cv2
  import os

  # 创建输出文件夹（如果不存在）
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # 打开视频文件
  cap = cv2.VideoCapture(video_path)
  frame_count = 0

  while True:
    ret, frame = cap.read()
    if not ret:
      break
    # 保存每一帧为jpg文件
    cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count:04d}.jpg'), frame)
    frame_count += 1

  cap.release()
  print(f"总共提取了 {frame_count} 帧。")
  
video_path=r"test1.mp4"
output_folder=r".\output"
print("aa")
convert_mp4_to_jpg(video_path,output_folder)
