import os
import shutil
import random

def split_data(source_dirs, output_dirs, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    # 각각의 이미지 폴더(source_dirs)에서 이미지들을 랜덤으로 나누기
    for source_dir, output_dir in zip(source_dirs, output_dirs):
        if not os.path.isdir(source_dir):
            print(f"Error: {source_dir} is not a valid directory.")
            continue

        # 폴더 이름으로 라벨 지정
        label = os.path.basename(source_dir)
        images = os.listdir(source_dir)

        # 이미지가 없으면 경고
        if not images:
            print(f"Warning: No images found in {source_dir}")
            continue

        # 이미지를 랜덤하게 섞음
        random.shuffle(images)

        # 이미지 개수 계산
        total_images = len(images)
        train_count = int(total_images * train_ratio)
        test_count = int(total_images * test_ratio)
        valid_count = total_images - train_count - test_count

        # 데이터 나누기
        train_images = images[:train_count]
        test_images = images[train_count:train_count + test_count]
        valid_images = images[train_count + test_count:]

        # 출력 폴더 내에 train, test, valid 폴더 생성
        train_dir = os.path.join(output_dir, 'train', label)
        test_dir = os.path.join(output_dir, 'test', label)
        valid_dir = os.path.join(output_dir, 'valid', label)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)

        # 이미지 이동
        for image in train_images:
            shutil.move(os.path.join(source_dir, image), os.path.join(train_dir, image))
        for image in test_images:
            shutil.move(os.path.join(source_dir, image), os.path.join(test_dir, image))
        for image in valid_images:
            shutil.move(os.path.join(source_dir, image), os.path.join(valid_dir, image))

        print(f"Processed {label}: train {train_count}, test {test_count}, valid {valid_count} images.")

# 원본 이미지 폴더들이 들어있는 디렉토리 목록
source_dirs = [
    '../dataset/crop_roi2/stop',
    '../dataset/crop_roi2/corner_right'
]

# 각 결과를 저장할 디렉토리 목록 (각각의 폴더에 결과를 따로 저장)
output_dirs = [
    '../dataset/split_dataset/stop',
    '../dataset/split_dataset/corner_right'
]

# 데이터 분할 실행 (70% train, 20% test, 10% valid 비율로 나누기)
split_data(source_dirs, output_dirs)
