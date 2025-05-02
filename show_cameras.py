#!/usr/bin/env python

import cv2
import numpy as np
from pathlib import Path
import glob
import re

def natural_sort_key(s):
    """숫자를 포함한 문자열의 자연스러운 정렬을 위한 키 함수"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def main():
    # 이미지 디렉토리
    root_dir = Path('/home/ctrl2/Bench2Drive/data_collect_output/camera')
    
    # 카메라 배치 순서 (2x3 그리드)
    camera_layout = [
        ['rgb_front_left', 'rgb_front', 'rgb_front_right'],
        ['rgb_back_left', 'rgb_back', 'rgb_back_right']
    ]
    
    # 각 카메라별 이미지 파일 목록
    camera_images = {}
    for camera in sum(camera_layout, []):
        image_files = glob.glob(str(root_dir / camera / '*.jpg'))
        image_files.sort(key=natural_sort_key)
        camera_images[camera] = image_files
    
    # 모든 카메라의 프레임 수 확인
    min_frames = min(len(files) for files in camera_images.values())
    if min_frames == 0:
        print("이미지 파일을 찾을 수 없습니다!")
        return
    
    print(f"총 {min_frames}개의 프레임을 표시합니다.")
    print("\n조작 방법:")
    print("q: 종료")
    print("a: 이전 프레임")
    print("d: 다음 프레임")
    print("r: 처음으로")
    print("s: 재생 속도 변경")
    print("space: 일시정지/재생")
    
    frame_idx = 0
    frame_skip = 1  # 프레임 스킵 수 (1: 모든 프레임, 2: 2프레임마다, ...)
    wait_time = 1  # 프레임 간 대기 시간 (ms)
    is_playing = True
    
    while True:
        # 2x3 그리드 이미지 생성
        grid = []
        for row in camera_layout:
            grid_row = []
            for camera in row:
                if frame_idx < len(camera_images[camera]):
                    img = cv2.imread(camera_images[camera][frame_idx])
                    # 이미지 크기 조정 (가로 320, 세로 180)
                    img = cv2.resize(img, (320, 180))
                    # 카메라 이름 표시
                    cv2.putText(img, camera.replace('rgb_', ''), (5, 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    grid_row.append(img)
                else:
                    # 빈 이미지
                    grid_row.append(np.zeros((180, 320, 3), dtype=np.uint8))
            grid.append(np.hstack(grid_row))
        
        # 모든 행 합치기
        full_grid = np.vstack(grid)
        
        # 상태 정보 표시
        status = f"Frame: {frame_idx} | Speed: {1000/wait_time:.1f}fps | Skip: {frame_skip}"
        if not is_playing:
            status += " | PAUSED"
        cv2.putText(full_grid, status, (5, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 이미지 표시
        cv2.imshow('Camera Views', full_grid)
        
        # 키 입력 처리
        key = cv2.waitKey(wait_time)  # 대기 시간
        if key == ord('q'):  # q 키로 종료
            break
        elif key == ord('a'):  # a 키로 이전 프레임
            frame_idx = max(0, frame_idx - frame_skip)
        elif key == ord('d'):  # d 키로 다음 프레임
            frame_idx = min(min_frames - 1, frame_idx + frame_skip)
        elif key == ord('r'):  # r 키로 처음으로
            frame_idx = 0
        elif key == ord('s'):  # s 키로 재생 속도 변경
            if wait_time == 1:
                wait_time = 5
                frame_skip = 1
            elif wait_time == 5:
                wait_time = 10
                frame_skip = 2
            elif wait_time == 10:
                wait_time = 1
                frame_skip = 5
            else:
                wait_time = 1
                frame_skip = 1
        elif key == ord(' '):  # 스페이스바로 일시정지/재생
            is_playing = not is_playing
        elif is_playing:  # 자동 재생
            frame_idx = (frame_idx + frame_skip) % min_frames
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 