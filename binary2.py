from PIL import Image
import os

# 처리할 이미지 파일들이 있는 폴더 경로 리스트
folders_list = [


    "../dataset/test8/corner_right"



]

# 이진화된 이미지를 저장할 폴더 리스트 (각 입력 폴더에 대응)
output_folders = [


    "../dataset/test8_binary2/corner_right"



]

# 각 폴더에 대해 작업 수행
threshold_value = 165  # 임계값을 180으로 높여서 밝은 이미지 처리
for folder, output_folder in zip(folders_list, output_folders):
    # 저장할 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 해당 폴더 내 모든 파일에 대해 작업
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # 이미지 파일 경로
            image_path = os.path.join(folder, filename)

            try:
                # 이미지 파일 열기
                print(f"처리 중인 파일: {image_path}")  # 파일 경로 출력
                image = Image.open(image_path)

                # 이미지를 흑백으로 변환
                grayscale_image = image.convert("L")  # "L" 모드는 흑백 (grayscale)

                # 이진화 처리 (임계값 조정)
                binary_image = grayscale_image.point(lambda p: 0 if p < threshold_value else 255, mode='1')

                # 자른 이미지 저장 경로 (새로운 폴더에 저장)
                binary_image_path = os.path.join(output_folder, f"binary_{filename}")

                # 이진화된 이미지 저장
                binary_image.save(binary_image_path)

                print(f"{filename} 흑백 변환 및 이진화 후 {binary_image_path}에 저장 완료.")
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {filename}, 오류: {e}")

print("모든 이미지 흑백 변환 및 이진화 저장이 완료되었습니다.")
