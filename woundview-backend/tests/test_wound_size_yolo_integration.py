import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import os
from wound_size_yolo import estimate_wound_size

YOLO_MODEL_PATH = "/Users/hanseung-yeon/Desktop/woundview-fb/woundview-backend/yolov8.pt"  # 실제 파일명/경로로 수정 필요
SAM_MODEL_PATH = "/Users/hanseung-yeon/Desktop/woundview-fb/woundview-backend/sam_vit_h.pth"  # 실제 파일명/경로로 수정 필요

def test_full_pipeline_success():
    image_path = "/Users/hanseung-yeon/Desktop/Wound_dataset/Bruises/bruises (20).jpg"
    assert os.path.exists(image_path)
    assert os.path.exists(YOLO_MODEL_PATH)
    assert os.path.exists(SAM_MODEL_PATH)
    result = estimate_wound_size(image_path, YOLO_MODEL_PATH, SAM_MODEL_PATH)
    # 반환값이 dict라면 아래처럼, 아니면 타입/값에 맞게 수정
    assert isinstance(result, dict)
    assert "wound_size_mm" in result

def test_image_not_found():
    with pytest.raises(ValueError):
        estimate_wound_size("/invalid/path.jpg", YOLO_MODEL_PATH, SAM_MODEL_PATH)

def test_yolo_model_not_found():
    image_path = "/Users/hanseung-yeon/Desktop/Wound_dataset/Bruises/bruises (20).jpg"
    with pytest.raises(FileNotFoundError):
        estimate_wound_size(image_path, "/invalid/yolo.pt", SAM_MODEL_PATH)

def test_sam_model_not_found():
    image_path = "/Users/hanseung-yeon/Desktop/Wound_dataset/Bruises/bruises (20).jpg"
    with pytest.raises(FileNotFoundError):
        estimate_wound_size(image_path, YOLO_MODEL_PATH, "/invalid/sam.pt")
