from PIL import Image
import os
import random

# 디렉토리 리스트 설정
folder9 = ['./train4/straight',
           './train4/left',
           './train4/right',
           './train4/stop']

folder2 = ['./train8_binary/straight',
           './train8_binary/soft_left',
           './train8_binary/hard_left',
           './train8_binary/corner_left',
           './train8_binary/soft_right',
           './train8_binary/hard_right',
           './train8_binary/corner_right',
           './train8_binary/stop']

folder3 = ['./train6_binary/straight',
           './train6_binary/soft_left',
           './train6_binary/hard_left',
           './train6_binary/soft_right',
           './train6_binary/hard_right',
           './train6_binary/stop']

# 디렉토리 내에서 이미지를 선택하고 병합하는 함수
def merge_class_images(class_folders, output_file="merged_image.png"):
    images_per_class = []

    for folder in class_folders:
        if os.path.exists(folder):
            # 해당 디렉토리 내 이미지 파일 목록 가져오기
            image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                # 이미지 파일 중 하나를 무작위로 선택
                image_file = random.choice(image_files)
                image_path = os.path.join(folder, image_file)
                img = Image.open(image_path)
                images_per_class.append(img)
            else:
                print(f"No images found in {folder}")
        else:
            print(f"Path {folder} does not exist.")

    # 각 클래스의 이미지들이 있으면 병합 시작
    if images_per_class:
        # 이미지를 가로로 병합 (각 이미지는 같은 크기라고 가정)
        widths, heights = zip(*(img.size for img in images_per_class))
        total_width = sum(widths)
        max_height = max(heights)

        # 새 이미지 캔버스 생성
        merged_image = Image.new('RGB', (total_width, max_height))

        # 각 이미지를 병합된 큰 이미지에 붙여넣음
        x_offset = 0
        for img in images_per_class:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # 병합된 이미지를 파일로 저장
        merged_image.save(output_file)
        print(f"Image saved as {output_file}")
    else:
        print("No images to merge.")

# 함수 실행 - 각 폴더별로 하나의 이미지를 만들어 냄
merge_class_images(folder9, output_file="output_folder9.png")
merge_class_images(folder2, output_file="output_folder2.png")
merge_class_images(folder3, output_file="output_folder3.png")
