from PIL import Image
import os

# 5개의 이미지 폴더 경로
folders = [
    "../dataset/test_capture/straight",
    "../dataset/test_capture/soft_left",
    "../dataset/test_capture/hard_left",
    "../dataset/test_capture/corner_left",
    "../dataset/test_capture/soft_right",
    "../dataset/test_capture/hard_right",
    "../dataset/test_capture/corner_right",
]

# 자른 이미지를 저장할 경로 (각 폴더별로 저장)
output_folders = [
    "../dataset/test8/straight",
    "../dataset/test8/soft_left",
    "../dataset/test8/hard_left",
    "../dataset/test8/corner_left",
    "../dataset/test8/soft_right",
    "../dataset/test8/hard_right",
    "../dataset/test8/corner_right",
]

# 각 폴더별로 작업 수행
for folder, output_folder in zip(folders, output_folders):
    # 저장할 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 폴더 내의 모든 파일에 대해 작업
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # 이미지 파일 경로
            image_path = os.path.join(folder, filename)

            # 이미지 파일 열기
            image = Image.open(image_path)

            # 이미지 크기 확인
            width, height = image.size

            # 아래에서부터 세로로 1/2만큼 자르기
            left = 0
            top = height // 2  # 세로 크기의 아래 1/2 시작점
            right = width
            bottom = height  # 이미지의 맨 아래까지

            cropped_image = image.crop((left, top, right, bottom))

            # 자른 이미지 저장 경로
            cropped_image_path = os.path.join(output_folder, filename)

            # 자른 이미지 저장
            cropped_image.save(cropped_image_path)

            print(f"{filename} 자르고 {cropped_image_path}에 저장 완료.")

print("모든 이미지 자르기 및 저장이 완료되었습니다.")
