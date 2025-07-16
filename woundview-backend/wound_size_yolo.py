import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from skimage.morphology import skeletonize, medial_axis # 스켈레톤 추출을 위함
from scipy.spatial.distance import euclidean # 스켈레톤 경로 거리 계산을 위함
import matplotlib.pyplot as plt # 시각화를 위함 (Colab에서 이미지 표시)


#헬퍼함수 정의----------------------------------------------------------------------------------
def calculate_mm_per_px(card_bbox_pixels, actual_card_width_mm=85.6, actual_card_height_mm=53.98):
    """
    신용카드 바운딩 박스를 기준으로 픽셀당 밀리미터 비율을 계산합니다.
    Args:
        card_bbox_pixels (list): [x1, y1, x2, y2] 형식의 카드 바운딩 박스 픽셀 좌표.
        actual_card_width_mm (float): 신용카드의 실제 가로 길이 (mm).
        actual_card_height_mm (float): 신용카드의 실제 세로 길이 (mm).
    Returns:
        float: 픽셀당 밀리미터 비율 (mm/px) 또는 None (계산 불가 시).
    """
    x1, y1, x2, y2 = card_bbox_pixels
    pixel_width = x2 - x1
    pixel_height = y2 - y1

    if pixel_width > 0 and pixel_height > 0:
        # 가로 또는 세로 중 더 긴 쪽을 기준으로 비율 계산 (안정성 증가)
        if pixel_width / actual_card_width_mm > pixel_height / actual_card_height_mm:
            ratio = actual_card_width_mm / pixel_width
        else:
            ratio = actual_card_height_mm / pixel_height
        return ratio
    return None

def calculate_longest_skeleton_path_length(mask_image_uint8, mm_per_px_ratio):
    """
    이진 마스크에서 스켈레톤을 추출하고 가장 긴 경로의 길이를 계산합니다.
    Args:
        mask_image_uint8 (numpy.array): 0 또는 255 값으로 이루어진 상처 마스크 (uint8).
        mm_per_px_ratio (float): 픽셀당 밀리미터 비율.
    Returns:
        tuple: (가장 긴 경로의 실제 길이(mm), 해당 경로의 시작 및 끝 좌표 (x1, y1, x2, y2)).
    """
    if np.sum(mask_image_uint8) == 0:
        return 0, (0,0,0,0)

    # 스켈레톤화 (마스크를 얇은 선으로 만듦)
    skeleton = skeletonize(mask_image_uint8 > 0)

    # 스켈레톤의 끝점 찾기
    endpoints = []
    for r, c in np.argwhere(skeleton):
        neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1] and skeleton[nr, nc]:
                    neighbors += 1
        if neighbors == 1: # 이웃이 1개면 끝점
            endpoints.append((r, c))

    if len(endpoints) < 2: # 끝점이 2개 미만이면 유효한 경로를 찾기 어려움
        return 0, (0,0,0,0)

    max_path_length_px = 0
    longest_path_coords = (0,0,0,0)

    # 모든 끝점 쌍 사이의 유클리드 거리를 계산하여 가장 긴 경로 선택 (근사치)
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            p1 = endpoints[i]
            p2 = endpoints[j]
            dist_px = euclidean(p1, p2)
            if dist_px > max_path_length_px:
                max_path_length_px = dist_px
                # (x1, y1), (x2, y2) 순서로 저장
                longest_path_coords = (p1[1], p1[0], p2[1], p2[0])

    return max_path_length_px * mm_per_px_ratio, longest_path_coords
#--------------------------------------------------------------------------------------------------------------------------------------

def estimate_wound_size(original_image_path):
    wound_actual_width_mm = 0
    wound_actual_height_mm = 0
    wound_actual_area_mm2 = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용할 장치: {device}")

    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_YELLOW = (0, 255, 255)

    # 파일 경로 (코랩 환경에 맞게 업로드 후 사용)
    # TODO: 당신의 YOLOv8 학습 모델 파일명으로 변경
    YOLO_MODEL_PATH = 'my_trained_model.pt'

    # SAM 모델 타입 ('vit_l' 또는 'vit_b' 권장, 'vit_h'는 메모리 소모 큼)
    SAM_MODEL_TYPE = 'vit_l'

    # TODO: sam_vit_l_0b3195.pth 또는 sam_vit_b_01ec64.pth 파일을 Google Drive에 업로드 후 경로 수정!
    # 다운로드 링크:
    # 'vit_l': https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    # 'vit_b': https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    SAM_CHECKPOINT_PATH =  'sam_vit_l_0b3195.pth' # <-- 이 경로를 당신의 구글 드라이브 경로로 변경

    # TODO: 분석할 이미지 파일명 또는 경로로 변경
    # 만약 콜랩 세션에 직접 업로드했다면 파일명만, 드라이브에 있다면 전체 경로를 입력
    IMAGE_PATH = original_image_path

    # 탐지 플래그 초기화
    found_card = False
    found_wound = False
    mm_per_px = None # 픽셀당 밀리미터 비율 초기화

    # display_img 초기화 (이미지 로드 전에 사용될 경우를 대비)
    display_img = np.zeros((720, 1280, 3), dtype=np.uint8) # 임시 빈 이미지, 실제 해상도는 이미지 로드 후 결정

    # 시각화 오버레이 투명도
    MASK_OVERLAY_ALPHA = 0.5

    # ================================================================
    # --- 4. 메인 프로세스 ---
    # ================================================================
    print("\n--- 상처 길이 측정 프로그램 시작 ---")

    try:
        # 4.1. Google Drive 마운트 (로컬 환경에서는 불필요하므로 제거)
        # print("Google Drive 마운트 시도...")
        # drive.mount('/content/drive')
        # print("✅ Google Drive 마운트 완료.")
        # (로컬 환경에서는 Google Drive 마운트 필요 없음)


        # 4.2. 입력 이미지 로드 및 초기 설정
        print(f"'{IMAGE_PATH}' 이미지 로드 중...")
        if not os.path.exists(IMAGE_PATH):
            raise FileNotFoundError(f"'{IMAGE_PATH}' 파일을 현재 Colab 환경에서 찾을 수 없습니다. 파일을 업로드했는지, 경로가 올바른지 확인해주세요.")

        img_bgr = cv2.imread(IMAGE_PATH)
        if img_bgr is None:
            raise ValueError(f"'{IMAGE_PATH}' 이미지를 로드할 수 없습니다. 파일이 손상되었거나 이미지가 아닌지 확인해주세요.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_height, original_width, _ = img_rgb.shape
        print(f"✅ 원본 이미지 로드 완료. 해상도: {original_width}x{original_height}")
    except (FileNotFoundError, ValueError) as e:
        print(f"에러 발생: {e}")
        return None

        # 이미지 해상도 조정 (메모리 절약 및 처리 속도 향상)
        max_dim = 1024 # 이미지의 최대 가로/세로 길이
        if max(original_height, original_width) > max_dim:
            scale_factor = max_dim / max(original_height, original_width)
            new_w = int(original_width * scale_factor)
            new_h = int(original_height * scale_factor)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"이미지 해상도 조정: {original_width}x{original_height} -> {new_w}x{new_h}")
            display_img = np.zeros((new_h, new_w, 3), dtype=np.uint8) # display_img 해상도 업데이트
        else:
            display_img = np.zeros((original_height, original_width, 3), dtype=np.uint8) # display_img 초기화

        display_img[:, :, :] = img_rgb[:, :, :] # 원본 이미지 display_img에 복사

        # 4.3. YOLO 모델 로드
        print(f"YOLO 모델 로드 중: '{YOLO_MODEL_PATH}'...")
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"'{YOLO_MODEL_PATH}' 파일을 현재 Colab 환경에서 찾을 수 없습니다. 파일을 업로드했는지, 경로가 올바른지 확인해주세요.")

        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model.to(device)
        print("✅ YOLO 모델 로드 완료.")

        # 4.4. SAM 모델 로드
        print(f"SAM 모델 로드 중: '{SAM_CHECKPOINT_PATH}' (타입: {SAM_MODEL_TYPE})...")
        if not os.path.exists(SAM_CHECKPOINT_PATH):
            raise FileNotFoundError(f"'{SAM_CHECKPOINT_PATH}' 파일을 현재 Colab 환경 또는 Google Drive에서 찾을 수 없습니다. 파일을 업로드했는지, 경로가 올바른지 확인해주세요.")

        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device)
        sam_predictor = SamPredictor(sam)
        print("✅ SAM 모델 로드 완료.")

        # GPU 메모리 캐시 비우기 (모델 로드 후, 최적화)
        if device == 'cuda':
            torch.cuda.empty_cache()
            print("✅ GPU 캐시 비우기 완료.")

        # 4.5. YOLO를 이용한 카드 및 상처 탐지
        print("YOLO 탐지 시작 (카드 및 상처)...")
        results = yolo_model(img_rgb, verbose=False)[0] # verbose=False로 콘솔 출력 줄임
        class_names = yolo_model.names # YOLO 모델의 클래스 이름 가져오기

        card_bbox_xyxy = None
        wound_bbox_xyxy = None
        found_card = False # found_card 변수 초기화
        found_wound = False # found_wound 변수 초기화 (추가)

        # 카드 탐지 및 픽셀당 밀리미터 비율 계산
        for box_idx in range(len(results.boxes)):
            class_id = int(results.boxes.cls[box_idx].item())
            obj_class_name = class_names[class_id]
            bbox_xyxy = results.boxes.xyxy[box_idx].cpu().numpy().astype(int)

            if obj_class_name == 'card': # TODO: 당신의 YOLO 모델 'card' 클래스 이름 확인
                found_card = True
                card_bbox_xyxy = bbox_xyxy
                print(f"✅ 카드 탐지 완료. YOLO BBox: {card_bbox_xyxy}")
                cv2.rectangle(display_img, (card_bbox_xyxy[0], card_bbox_xyxy[1]),
                            (card_bbox_xyxy[2], card_bbox_xyxy[3]), COLOR_GREEN, 2)

                mm_per_px = calculate_mm_per_px(card_bbox_xyxy)
                if mm_per_px:
                    cv2.putText(display_img, f"Scale: {mm_per_px:.4f} mm/px",
                                (card_bbox_xyxy[0], card_bbox_xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
                    print(f"✅ 픽셀당 밀리미터 비율 계산 완료: {mm_per_px:.4f} mm/px")
                else:
                    print("❌ mm_per_px 계산 실패: 카드 바운딩 박스 오류.")
                break # 첫 번째 카드만 사용 (여러 카드 모두 탐지하려면 제거)

        # 상처 탐지 및 SAM을 이용한 마스크 정밀화
        if found_card and mm_per_px is not None:
            wound_mask_refined = None
            # 상처를 찾기 위한 두 번째 반복문 (카드 탐지와 분리)
            for box_idx in range(len(results.boxes)):
                class_id = int(results.boxes.cls[box_idx].item())
                obj_class_name = class_names[class_id]

                # --- 'wound' 클래스 탐지 시 실행되는 블록 시작 ---
                if obj_class_name == 'wound': # TODO: 당신의 YOLO 모델 'wound' 클래스 이름 확인
                    found_wound = True
                    wound_bbox_xyxy = results.boxes.xyxy[box_idx].cpu().numpy().astype(int)

                    # 상처 바운딩 박스 추가 코드 (YOLO 탐지 결과 시각화)
                    cv2.rectangle(display_img, (wound_bbox_xyxy[0], wound_bbox_xyxy[1]),
                                (wound_bbox_xyxy[2], wound_bbox_xyxy[3]), COLOR_YELLOW, 2) # YELLOW 색상으로 2픽셀 두께의 사각형 그리기
                    cv2.putText(display_img, "Wound Box", (wound_bbox_xyxy[0], wound_bbox_xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2) # "Wound Box" 텍스트 추가

                    sam_predictor.set_image(img_rgb) # SAM 모델에 이미지 설정 (한 번만)

                    # YOLO 바운딩 박스를 SAM 프롬프트로 사용하여 마스크 정밀화
                    masks, scores, _ = sam_predictor.predict(
                        point_coords=None, box=wound_bbox_xyxy, multimask_output=False,
                    )

                    wound_mask_refined = masks[0] # (H, W) boolean mask
                    wound_mask_resized_uint8 = (wound_mask_refined * 255).astype(np.uint8)

                    print(f"✅ SAM 마스크 정밀화 완료. 마스크 픽셀 수: {np.sum(wound_mask_refined)}")

                    if device == 'cuda': # GPU 메모리 캐시 비우기 (SAM 예측 후)
                        torch.cuda.empty_cache()
                        print("✅ GPU 캐시 비우기 완료 (SAM 예측 후).")

                    break # 첫 번째 상처만 사용 (여러 상처 모두 측정하려면 이 줄을 제거하세요.)
                # --- 'wound' 클래스 탐지 시 실행되는 블록 끝 ---

            # 4.6. 상처 길이/너비/면적 계산 및 시각화
            # 이 블록은 상처 탐지 루프 (위의 for box_idx in range...)가 끝난 후에 실행되어야 합니다.
            # 즉, found_wound가 설정되고 wound_mask_refined가 최종적으로 결정된 후입니다.
            # 코드의 원래 들여쓰기에 따라 그대로 두었지만, 논리적으로는 이 for 루프 밖으로 나와야 합니다.
            # 현재는 for 루프 (wound 찾기)가 끝나면 바로 실행되도록 되어 있습니다.

            if wound_mask_refined is not None and mm_per_px is not None:
                wound_pixel_width_measured = 0
                wound_pixel_height_measured = 0
                wound_actual_width_mm = 0
                wound_actual_height_mm = 0
                wound_pixel_area = 0
                wound_actual_area_mm2 = 0

                # 상처 마스크에서 윤곽선 찾기 및 측정
                contours, _ = cv2.findContours(wound_mask_resized_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)

                    rect = cv2.minAreaRect(largest_contour) # 최소 면적 회전 사각형
                    (center_x, center_y), (width, height), angle = rect

                    wound_pixel_width_measured = width
                    wound_pixel_height_measured = height

                    wound_actual_width_mm = wound_pixel_width_measured * mm_per_px
                    wound_actual_height_mm = wound_pixel_height_measured * mm_per_px
                    wound_pixel_area = cv2.contourArea(largest_contour)
                    wound_actual_area_mm2 = wound_pixel_area * (mm_per_px ** 2)

                    print(f"상처 픽셀 너비 (마스크 기반): {wound_pixel_width_measured:.2f}px")
                    print(f"상처 픽셀 높이 (마스크 기반): {wound_pixel_height_measured:.2f}px")
                    print(f"상처 픽셀 면적 (마스크 기반): {wound_pixel_area:.2f}px^2")
                    print(f"추정된 상처 실제 너비: {wound_actual_width_mm:.2f} mm")
                    print(f"추정된 상처 실제 높이: {wound_actual_height_mm:.2f} mm")
                    print(f"추정된 상처 실제 면적: {wound_actual_area_mm2:.2f} mm^2")

                else:
                    # 윤곽선을 못 찾았을 때, 바운딩 박스 기반으로 계산 (정확도 낮음)
                    if wound_bbox_xyxy is not None:
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = wound_bbox_xyxy
                        wound_pixel_width_measured = bbox_x2 - bbox_x1
                        wound_pixel_height_measured = bbox_y2 - bbox_y1
                        wound_actual_width_mm = wound_pixel_width_measured * mm_per_px
                        wound_actual_height_mm = wound_pixel_height_measured * mm_per_px
                        wound_pixel_area = wound_pixel_width_measured * wound_pixel_height_measured
                        wound_actual_area_mm2 = wound_pixel_area * (mm_per_px ** 2)
                        print("SAM 마스크에서 윤곽선을 찾을 수 없어 바운딩 박스 기준으로 길이/면적 측정되었습니다. 정확도가 낮을 수 있습니다.")
                    else:
                        print("❌ 상처 바운딩 박스 정보도 없어 길이/면적 측정 불가.")

                # 스켈레톤 기반 상처 길이 계산
                wound_mm_length, wound_line_info = calculate_longest_skeleton_path_length(
                    wound_mask_resized_uint8, mm_per_px
                )
                print(f"✅ 스켈레톤 기반 상처 길이: {wound_mm_length:.2f} mm")

                # 상처 마스크 오버레이 시각화
                for c in range(3):
                    display_img[:, :, c] = np.where(wound_mask_refined,
                                                    display_img[:, :, c] * (1 - MASK_OVERLAY_ALPHA) + COLOR_BLUE[c] * MASK_OVERLAY_ALPHA,
                                                    display_img[:, :, c])

                # 결과 선 그리기
                if wound_mm_length > 0:
                    x1_w, y1_w, x2_w, y2_w = wound_line_info
                    cv2.line(display_img, (x1_w, y1_w), (x2_w, y2_w), COLOR_GREEN, 3)

                else:
                    # 길이가 측정되지 않았을 때
                    if wound_actual_width_mm > 0:
                        cv2.putText(display_img, "Status: Partially Measured", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)
                    else:
                        cv2.putText(display_img, "Status: Not Measured", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
            else: # 상처 마스크 또는 mm_per_px 정보 없음
                cv2.putText(display_img, "Status: No Wound Info for Measure", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
        else: # 카드 없거나 mm_per_px 없음 (mm_per_px가 없으면 found_card가 거짓이거나 계산이 안 된 경우)
            cv2.putText(display_img, "Status: No Credit Card Detected!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
            cv2.putText(display_img, "Status: Scaling Failed (No Card)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)

# 상처 탐지 및 SAM을 이용한 마스크 정밀화
    if found_card and mm_per_px is not None:
        wound_mask_refined = None
        # 상처를 찾기 위한 두 번째 반복문 (카드 탐지와 분리)
        for box_idx in range(len(results.boxes)):
            class_id = int(results.boxes.cls[box_idx].item())
            obj_class_name = class_names[class_id]

            # --- 'wound' 클래스 탐지 시 실행되는 블록 시작 ---
            if obj_class_name == 'wound': # TODO: 당신의 YOLO 모델 'wound' 클래스 이름 확인
                found_wound = True
                wound_bbox_xyxy = results.boxes.xyxy[box_idx].cpu().numpy().astype(int)

                # 상처 바운딩 박스 추가 코드 (YOLO 탐지 결과 시각화)
                cv2.rectangle(display_img, (wound_bbox_xyxy[0], wound_bbox_xyxy[1]),
                            (wound_bbox_xyxy[2], wound_bbox_xyxy[3]), COLOR_YELLOW, 2) # YELLOW 색상으로 2픽셀 두께의 사각형 그리기
                cv2.putText(display_img, "Wound Box", (wound_bbox_xyxy[0], wound_bbox_xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2) # "Wound Box" 텍스트 추가

                sam_predictor.set_image(img_rgb) # SAM 모델에 이미지 설정 (한 번만)

                # YOLO 바운딩 박스를 SAM 프롬프트로 사용하여 마스크 정밀화
                masks, scores, _ = sam_predictor.predict(
                    point_coords=None, box=wound_bbox_xyxy, multimask_output=False,
                )

                wound_mask_refined = masks[0] # (H, W) boolean mask
                wound_mask_resized_uint8 = (wound_mask_refined * 255).astype(np.uint8)

                print(f"✅ SAM 마스크 정밀화 완료. 마스크 픽셀 수: {np.sum(wound_mask_refined)}")

                if device == 'cuda': # GPU 메모리 캐시 비우기 (SAM 예측 후)
                    torch.cuda.empty_cache()
                    print("✅ GPU 캐시 비우기 완료 (SAM 예측 후).")

            # --- 'wound' 클래스 탐지 시 실행되는 블록 끝 ---

        # 4.6. 상처 길이/너비/면적 계산 및 시각화
        # 이 블록은 상처 탐지 루프 (위의 for box_idx in range...)가 끝난 후에 실행되어야 합니다.
        # 즉, found_wound가 설정되고 wound_mask_refined가 최종적으로 결정된 후입니다.
        # 코드의 원래 들여쓰기에 따라 그대로 두었지만, 논리적으로는 이 for 루프 밖으로 나와야 합니다.
        # 현재는 for 루프 (wound 찾기)가 끝나면 바로 실행되도록 되어 있습니다.

        if wound_mask_refined is not None and mm_per_px is not None:
            wound_pixel_width_measured = 0
            wound_pixel_height_measured = 0
            wound_actual_width_mm = 0
            wound_actual_height_mm = 0
            wound_pixel_area = 0
            wound_actual_area_mm2 = 0

            # 상처 마스크에서 윤곽선 찾기 및 측정
            contours, _ = cv2.findContours(wound_mask_resized_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                rect = cv2.minAreaRect(largest_contour) # 최소 면적 회전 사각형
                (center_x, center_y), (width, height), angle = rect

                wound_pixel_width_measured = width
                wound_pixel_height_measured = height

                wound_actual_width_mm = wound_pixel_width_measured * mm_per_px
                wound_actual_height_mm = wound_pixel_height_measured * mm_per_px
                wound_pixel_area = cv2.contourArea(largest_contour)
                wound_actual_area_mm2 = wound_pixel_area * (mm_per_px ** 2)

                print(f"상처 픽셀 너비 (마스크 기반): {wound_pixel_width_measured:.2f}px")
                print(f"상처 픽셀 높이 (마스크 기반): {wound_pixel_height_measured:.2f}px")
                print(f"상처 픽셀 면적 (마스크 기반): {wound_pixel_area:.2f}px^2")
                print(f"추정된 상처 실제 너비: {wound_actual_width_mm:.2f} mm")
                print(f"추정된 상처 실제 높이: {wound_actual_height_mm:.2f} mm")
                print(f"추정된 상처 실제 면적: {wound_actual_area_mm2:.2f} mm^2")

            else:
                # 윤곽선을 못 찾았을 때, 바운딩 박스 기반으로 계산 (정확도 낮음)
                if wound_bbox_xyxy is not None:
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = wound_bbox_xyxy
                    wound_pixel_width_measured = bbox_x2 - bbox_x1
                    wound_pixel_height_measured = bbox_y2 - bbox_y1
                    wound_actual_width_mm = wound_pixel_width_measured * mm_per_px
                    wound_actual_height_mm = wound_pixel_height_measured * mm_per_px
                    wound_pixel_area = wound_pixel_width_measured * wound_pixel_height_measured
                    wound_actual_area_mm2 = wound_pixel_area * (mm_per_px ** 2)
                    print("SAM 마스크에서 윤곽선을 찾을 수 없어 바운딩 박스 기준으로 길이/면적 측정되었습니다. 정확도가 낮을 수 있습니다.")
                else:
                    print("❌ 상처 바운딩 박스 정보도 없어 길이/면적 측정 불가.")

            # 스켈레톤 기반 상처 길이 계산
            wound_mm_length, wound_line_info = calculate_longest_skeleton_path_length(
                wound_mask_resized_uint8, mm_per_px
            )
            print(f"✅ 스켈레톤 기반 상처 길이: {wound_mm_length:.2f} mm")

            # 상처 마스크 오버레이 시각화
            for c in range(3):
                display_img[:, :, c] = np.where(wound_mask_refined,
                                                display_img[:, :, c] * (1 - MASK_OVERLAY_ALPHA) + COLOR_BLUE[c] * MASK_OVERLAY_ALPHA,
                                                display_img[:, :, c])

            # 결과 선 그리기
            if wound_mm_length > 0:
                x1_w, y1_w, x2_w, y2_w = wound_line_info
                cv2.line(display_img, (x1_w, y1_w), (x2_w, y2_w), COLOR_GREEN, 3)

            else:
                # 길이가 측정되지 않았을 때
                if wound_actual_width_mm > 0:
                    cv2.putText(display_img, "Status: Partially Measured", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)
                else:
                    cv2.putText(display_img, "Status: Not Measured", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
        else: # 상처 마스크 또는 mm_per_px 정보 없음
            cv2.putText(display_img, "Status: No Wound Info for Measure", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
    else: # 카드 없거나 mm_per_px 없음 (mm_per_px가 없으면 found_card가 거짓이거나 계산이 안 된 경우)
        cv2.putText(display_img, "Status: No Credit Card Detected!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
        cv2.putText(display_img, "Status: Scaling Failed (No Card)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)

    try:
        cv2_imshow(display_img)
    except NameError:
        pass

    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {e}")
        # display_img가 정의되지 않았을 수 있으므로, 여기서도 확인
        if 'display_img' in locals():
            cv2.putText(display_img, f"Error: File Not Found! {e}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
            try:
                cv2_imshow(display_img)
            except NameError:
                pass
        else:
            print("이미지 로드 전 파일 오류 발생. 이미지를 표시할 수 없습니다.")
    except ValueError as e:
        print(f"❌ 데이터 오류: {e}")
        if 'display_img' in locals():
            cv2.putText(display_img, f"Error: Data Issue! {e}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
            try:
                cv2_imshow(display_img)
            except NameError:
                pass
        else:
            print("이미지 로드 중 데이터 오류 발생. 이미지를 표시할 수 없습니다.")
    except RuntimeError as e:
        print(f"❌ 런타임 오류: {e}")
        if 'display_img' in locals():
            cv2.putText(display_img, f"Error: Runtime Problem! {e}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
            try:
                cv2_imshow(display_img)
            except NameError:
                pass
        else:
            print("런타임 오류 발생. 이미지를 표시할 수 없습니다.")
    except Exception as e:
        print(f"❌ 알 수 없는 오류 발생: {e}")
        if 'display_img' in locals():
            cv2.putText(display_img, f"Error: Unknown! {e}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
            try:
                cv2_imshow(display_img)
            except NameError:
                pass
        else:
            print("알 수 없는 오류 발생. 이미지를 표시할 수 없습니다.")

    print("\n--- 상처 길이 측정 프로그램 종료 ---")

    return {
        'wound_width': wound_actual_width_mm,
        'wound_height': wound_actual_height_mm,
        'wound_area': wound_actual_area_mm2
    }