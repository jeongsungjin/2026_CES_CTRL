import torch
import pytest
from .custom_bev_backbone import CustomBEVModel, BEVBackbone

def test_bev_backbone():
    # BEV 백본 테스트
    model = BEVBackbone(
        pretrained=True,
        backbone='resnet50',
        in_channels=3,
        out_channels=256,
        bev_size=(200, 200),
        output_size=(25, 25)
    )
    model.eval()
    
    # 테스트용 입력 생성
    dummy_input = torch.randn(1, 3, 200, 200)
    
    # 순전파
    with torch.no_grad():
        output = model(dummy_input)
    
    # 출력 형태 확인
    assert output.shape == (1, 256, 25, 25), f"BEVBackbone: Expected shape (1, 256, 25, 25), got {output.shape}"
    print("BEVBackbone test passed!")

def test_custom_bev_model():
    # 전체 모델 테스트
    model = CustomBEVModel(
        pretrained=True,
        backbone='resnet50',
        in_channels=3,
        out_channels=256,
        bev_size=(200, 200),
        output_size=(25, 25),
        num_encoder_layers=3
    )
    model.eval()
    
    # 테스트용 입력 생성
    dummy_input = torch.randn(1, 3, 200, 200)
    
    # 순전파
    with torch.no_grad():
        output = model(dummy_input)
    
    # 출력 형태 확인
    assert output.shape == (1, 256, 25, 25), f"CustomBEVModel: Expected shape (1, 256, 25, 25), got {output.shape}"
    print("CustomBEVModel test passed!")

if __name__ == "__main__":
    test_bev_backbone()
    test_custom_bev_model() 