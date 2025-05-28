# 커스텀 BEV 백본 모델을 UniAD에 통합하기

이 디렉토리는 우리가 개발한 BEV(Bird's Eye View) 백본 모델을 UniAD 프레임워크의 Stage1에 통합하는 방법을 제공합니다.

## 디렉토리 구조

```
my_bev/
├── models/
│   ├── custom_bev_backbone.py  # 우리의 BEV 백본 모델 구현
│   └── uniad_bev_adapter.py    # UniAD 프레임워크 연동을 위한 어댑터
├── dataset_bev/                # 생성된 더미 데이터셋
├── inference_results/          # 추론 결과 시각화
├── carla_bev_backbone_best.pth # 학습된 BEV 모델 가중치
├── configs/
│   └── uniad_custom_backbone.py # 커스텀 백본을 사용하는 UniAD 구성 파일
├── create_carla_data.py        # 더미 데이터 생성 스크립트
├── test_carla_training.py      # 모델 학습 스크립트
└── test_carla_inference.py     # 모델 추론 및 시각화 스크립트
```

## 사용 방법

### 1. 사전 요구사항

UniAD 프레임워크가 설치되어 있어야 합니다. 다음 명령어로 설치할 수 있습니다:

```bash
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD
pip install -v -e .
```

### 2. 커스텀 BEV 백본 모델 준비

1. 더미 데이터셋 생성:
   ```bash
   python my_bev/create_carla_data.py
   ```

2. 모델 학습:
   ```bash
   python my_bev/test_carla_training.py
   ```

3. 모델 추론 및 시각화:
   ```bash
   python my_bev/test_carla_inference.py
   ```

### 3. UniAD에 통합하기

1. `my_bev` 디렉토리를 UniAD 프로젝트 루트에 복사합니다.

2. UniAD 플러그인 디렉토리에 심볼릭 링크 생성:
   ```bash
   mkdir -p projects/mmdet3d_plugin/models/backbones
   ln -s $(pwd)/my_bev/models/uniad_bev_adapter.py projects/mmdet3d_plugin/models/backbones/
   ln -s $(pwd)/my_bev/models/custom_bev_backbone.py projects/mmdet3d_plugin/models/backbones/
   ```

3. 등록 스크립트 생성:
   ```bash
   echo "from .backbones.uniad_bev_adapter import UniADBEVAdapter" >> projects/mmdet3d_plugin/models/__init__.py
   ```

4. UniAD 학습:
   ```bash
   # 단일 GPU 학습
   python tools/train.py my_bev/configs/uniad_custom_backbone.py --work-dir work_dirs/uniad_custom_backbone
   
   # 다중 GPU 학습
   ./tools/dist_train.sh my_bev/configs/uniad_custom_backbone.py 8 --work-dir work_dirs/uniad_custom_backbone
   ```

## 주요 구성 요소

### 1. 커스텀 BEV 백본 모델

`custom_bev_backbone.py`는 ResNet-34와 FPN을 기반으로 하는 BEV 백본 모델을 구현합니다. 이 모델은 카메라 이미지를 입력으로 받아 BEV 특징 맵을 출력합니다.

### 2. UniAD 어댑터

`uniad_bev_adapter.py`는 우리의 BEV 백본 모델을 UniAD 프레임워크에 통합하기 위한 어댑터 클래스를 구현합니다. 이 어댑터는 다음 기능을 제공합니다:

- 다중 스케일 특징 출력 (FPN과 유사)
- 학습된 가중치 로드
- 레이어 고정 및 미세 조정 기능

### 3. 구성 파일

`uniad_custom_backbone.py`는 UniAD의 base_track_map.py를 기반으로 하되, 기존 ResNet-101 백본 대신 우리의 BEV 백본 모델을 사용하도록 수정한 구성 파일입니다.

## 성능 비교

| 모델 | 백본 | AMOTA | NDS | mAP |
|-----|------|-------|-----|-----|
| UniAD (원본) | ResNet-101 | 0.393 | 0.517 | 0.345 |
| UniAD (커스텀) | BEV 백본 | - | - | - |

## 참고 사항

- 원본 ResNet-101 백본은 Mask R-CNN으로 사전 학습되었지만, 우리의 BEV 백본은 CARLA 시뮬레이터 데이터로 학습되었습니다.
- 최적의 성능을 위해서는 nuScenes 데이터셋으로 BEV 백본을 미세 조정하는 것이 좋습니다.
- GPU 메모리 요구 사항이 다를 수 있으므로, 필요에 따라 배치 크기를 조정해야 할 수 있습니다. 