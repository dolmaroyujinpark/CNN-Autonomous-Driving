import os
import csv

# 이미지 있는 상위 폴더 경로
root_dirs = ['../dataset/test4/straight',
           '../dataset/test4/left',
           '../dataset/test4/right',
           '../dataset/test4/stop']

# CSV 파일의 경로를 지정하세요
csv_file = './4test.csv'

# CSV 파일을 생성하고 데이터를 기록
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "label"])  # CSV 파일의 헤더

    for root_dir in root_dirs:
        label = os.path.basename(root_dir)  # 폴더 이름을 라벨로 사용
        for image_file in os.listdir(root_dir):
            if image_file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(root_dir, image_file)
                writer.writerow([image_path, label])

print(f"CSV 파일이 '{csv_file}'에 성공적으로 저장되었습니다.")
