import cv2
import os

# 비디오 파일 경로 목록
video_paths = [
    "../dataset/output_14.mp4",
    "../dataset/output_15.mp4",
    "../dataset/output_16.mp4",
    "../dataset/output_17.mp4",
    "../dataset/output_18.mp4",
    "../dataset/output_19.mp4",
    "../dataset/output_21.mp4"
]

# 각 비디오 파일에 대한 출력 폴더 경로 목록
output_folders = [
    "../dataset/test_capture/straight",
    "../dataset/test_capture/soft_left",
    "../dataset/test_capture/soft_right",
    "../dataset/test_capture/hard_left",
    "../dataset/test_capture/hard_right",
    "../dataset/test_capture/corner_right",
    "../dataset/test_capture/corner_left",
    ]

frame_interval = 5 # 5프레임 마다 한번 씩 저장

# 각 비디오 파일에 대해 작업
for video_path, video_output_folder in zip(video_paths, output_folders):
    # 비디오 캡처 객체 생성
    vidcap = cv2.VideoCapture(video_path)

    # 저장할 디렉토리가 없으면 생성
    os.makedirs(video_output_folder, exist_ok=True)

    # 프레임을 읽고 이미지로 저장
    success, image = vidcap.read()
    count = 0
    frame_count = 0

    while success:
        # 5프레임마다 한 번씩 저장
        if frame_count % frame_interval == 0:
            cv2.imwrite(os.path.join(video_output_folder, f"{count:06d}.jpg"), image)
            count += 1

        # 다음 프레임 읽기
        success, image = vidcap.read()
        frame_count += 1

    print(f'{os.path.basename(video_path)}: 총 {count}장의 이미지가 {video_output_folder}에 저장되었습니다.')
