#!/usr/bin/env python

import carla
import cv2
import numpy as np
import os
import time
from pathlib import Path
import random

def main():
    # 저장 경로 설정
    save_root = Path('/home/ctrl2/Bench2Drive/data_collect_output/camera')
    
    # 카메라 뷰별 저장 디렉토리 생성
    camera_dirs = [
        'rgb_front', 'rgb_front_left', 'rgb_front_right',
        'rgb_back', 'rgb_back_left', 'rgb_back_right'
    ]
    
    for cam_dir in camera_dirs:
        (save_root / cam_dir).mkdir(parents=True, exist_ok=True)
    
    client = None
    world = None
    settings = None
    sensors = []
    vehicles = []
    ego_vehicle = None
    
    try:
        # CARLA 클라이언트 연결
        print("CARLA 서버에 연결 중...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)  # 타임아웃 증가
        world = client.get_world()
        
        # 현재 월드의 설정 가져오기
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        
        # 트래픽 매니저 설정
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_synchronous_mode(True)
        
        # 에고 차량 생성
        print("에고 차량 생성 중...")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('role_name', 'hero')
        
        # 스폰 포인트 선택 (랜덤)
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicles.append(ego_vehicle)
        
        # 자율주행 활성화
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        
        # 다른 차량들 생성
        print("다른 차량들 생성 중...")
        for i in range(30):  # 30대의 차량 생성
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            if vehicle_bp.has_attribute('role_name'):
                vehicle_bp.set_attribute('role_name', 'autopilot')
            spawn_point = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                vehicles.append(vehicle)
        
        print(f"에고 차량 생성됨: {ego_vehicle.id}")
        print(f"총 {len(vehicles)}대의 차량 생성됨")
        
        # 센서 정의
        camera_configs = [
            {'name': 'rgb_front', 'x': 0.8, 'y': 0.0, 'z': 1.6, 'yaw': 0.0},
            {'name': 'rgb_front_left', 'x': 0.27, 'y': -0.55, 'z': 1.6, 'yaw': -55.0},
            {'name': 'rgb_front_right', 'x': 0.27, 'y': 0.55, 'z': 1.6, 'yaw': 55.0},
            {'name': 'rgb_back', 'x': -2.0, 'y': 0.0, 'z': 1.6, 'yaw': 180.0},
            {'name': 'rgb_back_left', 'x': -0.32, 'y': -0.55, 'z': 1.6, 'yaw': -110.0},
            {'name': 'rgb_back_right', 'x': -0.32, 'y': 0.55, 'z': 1.6, 'yaw': 110.0}
        ]
        
        # 카메라 센서 생성 및 설치
        print("카메라 센서 설치 중...")
        for config in camera_configs:
            bp = blueprint_library.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '1600')
            bp.set_attribute('image_size_y', '900')
            bp.set_attribute('fov', '70')
            
            # 센서 위치 설정
            spawn_point = carla.Transform(
                carla.Location(x=config['x'], y=config['y'], z=config['z']),
                carla.Rotation(yaw=config['yaw'])
            )
            
            sensor = world.spawn_actor(bp, spawn_point, attach_to=ego_vehicle)
            sensors.append(sensor)
            
            # 이미지 저장 콜백 설정
            sensor.listen(lambda image, name=config['name']: 
                save_image(image, name, save_root))
        
        print("데이터 수집 시작... Ctrl+C로 중지")
        while True:
            world.tick()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n데이터 수집 중지...")
    except Exception as e:
        print(f"\n에러 발생: {str(e)}")
    finally:
        print("\n정리 작업 시작...")
        # 설정 복구
        if world is not None and settings is not None:
            settings.synchronous_mode = False
            world.apply_settings(settings)
        
        # 센서 제거
        print("센서 제거 중...")
        for sensor in sensors:
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        
        # 차량 제거
        print("차량 제거 중...")
        for vehicle in vehicles:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
        
        print("정리 완료")

def save_image(image, camera_name, save_root):
    """이미지를 저장하는 함수"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    
    # 이미지 저장
    filename = f"{save_root}/{camera_name}/frame_{image.frame:05d}.jpg"
    cv2.imwrite(filename, array)

if __name__ == '__main__':
    main()