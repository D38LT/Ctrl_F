import os
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import shutil
import numpy as np
from scipy.signal import savgol_filter
import random

print("CUDA 사용 가능:", torch.cuda.is_available())
print("현재 CUDA 버전:", torch.version.cuda)
print("GPU 모델:", torch.cuda.get_device_name(0))

def normalize_class_name(class_name):
    """클래스 이름 정규화: 공백 제거 및 언더스코어 정리"""
    # 여러 개의 공백과 언더스코어를 단일 언더스코어로 변경
    normalized = '_'.join(filter(None, class_name.replace(' ', '_').split('_')))
    return normalized

def get_existing_classes(base_dir):
    """기존 데이터셋의 클래스 목록과 인덱스를 가져오기"""
    classes_file = os.path.join(base_dir, 'classes.txt')
    if not os.path.exists(classes_file):
        return {}
    
    classes_dict = {}
    with open(classes_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            class_name = normalize_class_name(line.strip())
            if class_name:  # 빈 줄 무시
                classes_dict[class_name] = idx
    
    return classes_dict

def prepare_directories(base_dir):
    """디렉토리 구조 설정"""
    # train/test 디렉토리
    dirs = {
        'train': {
            'images': os.path.join(base_dir, 'train', 'images'),
            'labels': os.path.join(base_dir, 'train', 'labels')
        },
        'test': {
            'images': os.path.join(base_dir, 'test', 'images'),
            'labels': os.path.join(base_dir, 'test', 'labels')
        },
        'visualization': os.path.join(base_dir, 'visualization')
    }
    
    # 기본 디렉토리 생성
    for split in ['train', 'test']:
        for path in dirs[split].values():
            os.makedirs(path, exist_ok=True)
    
    os.makedirs(dirs['visualization'], exist_ok=True)
    
    return dirs

def save_class_info(base_dir, class_name):
    """새로운 클래스 정보 저장"""
    classes_file = os.path.join(base_dir, 'classes.txt')
    normalized_name = normalize_class_name(class_name)
    classes_dict = get_existing_classes(base_dir)
    
    if normalized_name not in classes_dict:
        with open(classes_file, 'a', encoding='utf-8') as f:
            f.write(f"{normalized_name}\n")

def get_class_index(base_dir, class_name):
    """클래스의 인덱스 반환"""
    normalized_name = normalize_class_name(class_name)
    classes_dict = get_existing_classes(base_dir)
    
    if not classes_dict:
        # 첫 번째 클래스인 경우
        save_class_info(base_dir, normalized_name)
        return 0
    
    if normalized_name not in classes_dict:
        # 새로운 클래스인 경우
        new_index = max(classes_dict.values()) + 1
        save_class_info(base_dir, normalized_name)
        classes_dict[normalized_name] = new_index
        return new_index
    
    # 기존 클래스인 경우
    return classes_dict[normalized_name]

def select_videos(videos_dir):
    """처리할 비디오 파일 선택"""
    video_files = [f for f in os.listdir(videos_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print("처리할 비디오 파일이 없습니다!")
        return []
    
    print("\n=== 사용 가능한 비디오 파일 ===")
    for idx, file in enumerate(video_files, 1):
        print(f"{idx}. {file}")
    
    while True:
        try:
            selection = input("\n처리할 비디오 번호를 선택하세요 (여러 개는 쉼표로 구분, 전체는 'all'): ").strip()
            
            if selection.lower() == 'all':
                return video_files
            
            indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_files = [video_files[idx] for idx in indices if 0 <= idx < len(video_files)]
            
            if not selected_files:
                print("올바른 번호를 선택해주세요.")
                continue
                
            return selected_files
            
        except (ValueError, IndexError):
            print("올바른 형식으로 입력해주세요.")

def process_videos(videos_dir, base_dir="yolo_dataset", target_frames=300):
    """선택된 비디오 처리"""
    existing_classes = get_existing_classes(base_dir)
    
    # 비디오 선택
    selected_videos = select_videos(videos_dir)
    
    if not selected_videos:
        print("처리할 비디오가 선택되지 않았습니다!")
        return
    
    print(f"\n총 {len(selected_videos)}개의 비디오 파일이 선택됨")
    
    for video_file in selected_videos:
        # 비디오 파일 경로 설정
        video_path = os.path.join(videos_dir, video_file)
        
        # 비디오 파일명에서 클래스명 추출 (확장자 제외)
        class_name = os.path.splitext(video_file)[0]
        normalized_class_name = normalize_class_name(class_name)
        
        if normalized_class_name in existing_classes:
            print(f"\n클래스 '{normalized_class_name}'는 이미 존재합니다.")
            while True:
                choice = input("기존 데이터셋에 추가하시겠습니까? (y/n): ").lower()
                if choice == 'n':
                    print(f"클래스 '{normalized_class_name}' 처리를 건너뜁니다.")
                    break
                elif choice == 'y':
                    print(f"\n=== 처리 중: {video_file} (클래스: {normalized_class_name}) ===")
                    print("기존 데이터셋에 추가합니다...")
                    try:
                        process_single_video(video_path, normalized_class_name, base_dir, target_frames)
                        print(f"클래스 '{normalized_class_name}' 데이터 추가 완료")
                    except Exception as e:
                        print(f"비디오 처리 중 오류 발생: {e}")
                    break
                else:
                    print("'y' 또는 'n'으로 답해주세요.")
            continue
        
        print(f"\n=== ��리 중: {video_file} (새로운 클래스: {normalized_class_name}) ===")
        
        try:
            process_single_video(video_path, normalized_class_name, base_dir, target_frames)
            save_class_info(base_dir, normalized_class_name)
            print(f"새로운 클래스 '{normalized_class_name}' 처리 완료")
        except Exception as e:
            print(f"비디오 처리 중 오류 발생: {e}")
            continue

def process_single_video(video_path, class_name, base_dir, target_frames):
    """단일 비디오 처리"""
    temp_dir = os.path.join(base_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    dirs = prepare_directories(base_dir)
    
    try:
        print("1. 프레임 추출 시작...")
        frames_info = extract_frames(video_path, target_frames)
        
        print("\n2. 객체 인식 시작...")
        frames_data = process_frames(frames_info, class_name, base_dir)
        
        print("\n3. 프레임 검증 중...")
        valid_frames, invalid_frames = validate_frames(frames_data)
        
        print("\n4. 최종 결과 저장 중...")
        # 프레임 번호 순서대로 정렬
        valid_frames.sort()
        
        # 연속된 프레임을 8:2로 분할 (순서 유지)
        split_idx = int(len(valid_frames) * 0.8)
        train_frames = valid_frames[:split_idx]
        test_frames = valid_frames[split_idx:]
        
        successful_frames = 0
        with tqdm(total=len(valid_frames), desc="결과 저장 중") as pbar:
            # 학습 데이터 저장
            for frame_num in train_frames:
                frame_data = frames_data[frame_num]
                if frame_data['bbox']:
                    # 프레임 번호를 파일명에 포함시켜 순서 유지
                    save_frame_result(frame_data, f"{class_name}_{frame_num:04d}", 
                                    base_dir, class_name, 'train', dirs)
                    successful_frames += 1
                pbar.update(1)
            
            # 테스트 데이터 저장
            for frame_num in test_frames:
                frame_data = frames_data[frame_num]
                if frame_data['bbox']:
                    save_frame_result(frame_data, f"{class_name}_{frame_num:04d}", 
                                    base_dir, class_name, 'test', dirs)
                    successful_frames += 1
                pbar.update(1)
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def save_frame_result(frame_data, frame_name, base_dir, class_name, split, dirs):
    """프레임 결과 저장"""
    # 클래스별 디렉토리 생성
    for subdir in ['images', 'labels']:
        class_dir = os.path.join(dirs[split][subdir], class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    vis_class_dir = os.path.join(dirs['visualization'], class_name)
    os.makedirs(vis_class_dir, exist_ok=True)
    
    img_path = os.path.join(dirs[split]['images'], class_name, f"{frame_name}.jpg")
    label_path = os.path.join(dirs[split]['labels'], class_name, f"{frame_name}.txt")
    vis_path = os.path.join(dirs['visualization'], class_name, f"{frame_name}.jpg")
    
    frame_data['pil_image'].save(img_path)
    
    vis_image = frame_data['pil_image'].copy()
    vis_image = draw_bbox(vis_image, frame_data['bbox'], str(frame_data['label']))
    vis_image.save(vis_path)
    
    save_yolo_labels(label_path, frame_data['bbox'], frame_data['label'],
                    frame_data['pil_image'].width, frame_data['pil_image'].height,
                    base_dir)

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# Florence 모델 로드 및 설정
model_path = "microsoft/Florence-2-base"

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation="sdpa", 
        trust_remote_code=True
    )

if torch.cuda.is_available():
    model.to("cuda")

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def save_yolo_labels(label_path, bbox, label, img_width, img_height, base_dir):
    """YOLO 형식으로 라벨 저장"""
    class_id = get_class_index(base_dir, label)
    
    x1, y1, x2, y2 = bbox
    
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Phrase Grounding 실행 함수
def run_caption_to_phrase_grounding(image, text_input):
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # GPU 사용 확인
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"현재 디바이스: {next(model.parameters()).device}")
    
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    
    if torch.cuda.is_available():
        model.to("cpu")
        torch.cuda.empty_cache()
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    
    if '<CAPTION_TO_PHRASE_GROUNDING>' in parsed_answer:
        result = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']
        if result and "bboxes" in result and "labels" in result:
            confidences = result.get("scores", [1.0] * len(result["bboxes"]))
            max_conf_idx = confidences.index(max(confidences))
            result["bboxes"] = [result["bboxes"][max_conf_idx]]
            result["labels"] = [result["labels"][max_conf_idx]]
            return result
    return None

def get_box_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def check_frame_sequence(frames_data, max_center_distance=200):
    valid_frames = []
    invalid_frames = []
    prev_center = None
    
    for frame_num, frame_info in sorted(frames_data.items()):
        if not frame_info['bbox']:
            invalid_frames.append(frame_num)
            continue
            
        current_center = get_box_center(frame_info['bbox'])
        
        if prev_center is None:
            valid_frames.append(frame_num)
        else:
            distance = ((current_center[0] - prev_center[0])**2 + 
                       (current_center[1] - prev_center[1])**2)**0.5
            
            if distance > max_center_distance:
                invalid_frames.append(frame_num)
            else:
                valid_frames.append(frame_num)
                prev_center = current_center
                
        if frame_num in valid_frames:
            prev_center = current_center
            
    return valid_frames, invalid_frames

def draw_bbox(image, bbox, label):
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bbox
    draw.rectangle([(left, top), (right, bottom)], outline='red', width=2)
    draw.text((left, top-20), label, fill='red')
    return image

def extract_frames(video_path, num_frames=1000):
    temp_dir = os.path.join(os.path.dirname(video_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 최소 프레 수 확인
    if total_frames < num_frames:
        num_frames = total_frames
    
    # 간격 계산 수정
    min_interval = max(1, total_frames // (num_frames * 2))
    max_interval = max(min_interval + 1, min(total_frames // 10, total_frames // num_frames * 2))
    
    frames_info = {}
    frame_count = 0
    last_motion = 0
    prev_frame = None
    
    with tqdm(total=num_frames, desc="프레임 추출 중") as pbar:
        while frame_count < num_frames and len(frames_info) < num_frames:
            try:
                interval = random.randint(min_interval, max_interval)
                frame_pos = min(last_motion + interval, total_frames - 1)
                
                if frame_pos >= total_frames:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                    
                # 프레임 저장
                frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                frames_info[frame_count] = frame_path
                frame_count += 1
                pbar.update(1)
                last_motion = frame_pos
                
            except Exception as e:
                print(f"프레임 처리 중 오류: {e}")
                continue
    
    cap.release()
    return frames_info

def calculate_motion_score(prev_frame, curr_frame):
    # 프레임 간 차이 계산
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_diff) / 255.0

def calculate_quality_score(frame):
    # 이미지 품질 평가
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = np.std(gray)
    return laplacian_var * (0.5 + abs(brightness - 128)/128) * contrast

def process_frames(frames_info, class_name, base_dir):
    # 배치 크기 설정
    BATCH_SIZE = 4  # GPU 메모리에 따라 조정 가능
    
    frames_data = {}
    batch_images = []
    batch_nums = []
    
    with tqdm(total=len(frames_info), desc="객체 인식 중") as pbar:
        # GPU에 모델 한 번만 로드
        if torch.cuda.is_available():
            model.to("cuda")
            
        for frame_num, frame_path in frames_info.items():
            pil_image = Image.open(frame_path)
            batch_images.append(pil_image)
            batch_nums.append(frame_num)
            
            if len(batch_images) == BATCH_SIZE:
                # 배치 처리
                results = process_batch(batch_images, class_name)
                
                # 결과 저장
                for i, (frame_num, result) in enumerate(zip(batch_nums, results)):
                    frames_data[frame_num] = {
                        'frame_path': frames_info[frame_num],
                        'bbox': result["bboxes"][0] if result else None,
                        'label': result["labels"][0] if result else None,
                        'pil_image': batch_images[i]
                    }
                
                batch_images = []
                batch_nums = []
                pbar.update(BATCH_SIZE)
        
        # 남은 이미지 처리
        if batch_images:
            results = process_batch(batch_images, class_name)
            for i, (frame_num, result) in enumerate(zip(batch_nums, results)):
                frames_data[frame_num] = {
                    'frame_path': frames_info[frame_num],
                    'bbox': result["bboxes"][0] if result else None,
                    'label': result["labels"][0] if result else None,
                    'pil_image': batch_images[i]
                }
            pbar.update(len(batch_images))
            
        if torch.cuda.is_available():
            model.to("cpu")
            torch.cuda.empty_cache()
    
    return frames_data

def process_batch(images, text_input):
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = task_prompt + text_input
    
    # 배치 입력 준비
    inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt", padding=True)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():  # 추론 시 메모리 사용 감소
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    
    results = []
    for text in generated_texts:
        parsed_answer = processor.post_process_generation(
            text, 
            task=task_prompt, 
            image_size=(images[0].width, images[0].height)
        )
        
        if '<CAPTION_TO_PHRASE_GROUNDING>' in parsed_answer:
            result = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']
            if result and "bboxes" in result and "labels" in result:
                confidences = result.get("scores", [1.0] * len(result["bboxes"]))
                max_conf_idx = confidences.index(max(confidences))
                result["bboxes"] = [result["bboxes"][max_conf_idx]]
                result["labels"] = [result["labels"][max_conf_idx]]
                results.append(result)
            else:
                results.append(None)
        else:
            results.append(None)
    
    return results

def analyze_motion_pattern(frames_data, window_size=5, velocity_threshold=150, acceleration_threshold=75):
    positions = []
    frame_numbers = []
    
    for frame_num, frame_info in sorted(frames_data.items()):
        if frame_info['bbox']:
            center = get_box_center(frame_info['bbox'])
            positions.append(center)
            frame_numbers.append(frame_num)
    
    if len(positions) < window_size:
        return [], list(frames_data.keys())
    
    positions = np.array(positions)
    
    smoothed_x = savgol_filter(positions[:, 0], window_size, 2)
    smoothed_y = savgol_filter(positions[:, 1], window_size, 2)
    
    velocity_x = np.gradient(smoothed_x)
    velocity_y = np.gradient(smoothed_y)
    velocities = np.sqrt(velocity_x**2 + velocity_y**2)
    
    acceleration_x = np.gradient(velocity_x)
    acceleration_y = np.gradient(velocity_y)
    accelerations = np.sqrt(acceleration_x**2 + acceleration_y**2)
    
    directions = np.arctan2(velocity_y, velocity_x)
    
    valid_frames = []
    invalid_frames = []
    
    for i, frame_num in enumerate(frame_numbers):
        is_valid = True
        
        if velocities[i] > velocity_threshold:
            is_valid = False
        
        if accelerations[i] > acceleration_threshold:
            is_valid = False
        
        if i > 0:
            direction_change = abs(directions[i] - directions[i-1])
            if direction_change > np.pi/2:
                is_valid = False
        
        if is_valid:
            valid_frames.append(frame_num)
        else:
            invalid_frames.append(frame_num)
    
    return valid_frames, invalid_frames

def analyze_box_size(frames_data, size_threshold=5.0):
    """
    박스 크기가 급격하게 변하는 프레임 검출
    size_threshold: 이전 프레임 대비 허용되는 최대 크기 비율
    """
    valid_frames = []
    invalid_frames = []
    prev_size = None
    
    for frame_num, frame_info in sorted(frames_data.items()):
        if not frame_info['bbox']:
            invalid_frames.append(frame_num)
            continue
            
        bbox = frame_info['bbox']
        current_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # width * height
        
        if prev_size is None:
            valid_frames.append(frame_num)
        else:
            size_ratio = max(current_size / prev_size, prev_size / current_size)
            if size_ratio > size_threshold:
                invalid_frames.append(frame_num)
            else:
                valid_frames.append(frame_num)
                
        if frame_num in valid_frames:
            prev_size = current_size
            
    return valid_frames, invalid_frames

def validate_frames(frames_data):
    center_valid, _ = check_frame_sequence(frames_data, max_center_distance=200)
    motion_valid, _ = analyze_motion_pattern(frames_data)
    size_valid, _ = analyze_box_size(frames_data)
    
    print(f"중심점 검사 통과 프레임 수: {len(center_valid)}")
    print(f"움직임 패턴 검사 통과 프레임 수: {len(motion_valid)}")
    print(f"박스 크기 검사 통과 프레임 수: {len(size_valid)}")
    
    frame_validation_count = {}
    for frame in frames_data.keys():
        count = 0
        if frame in center_valid: count += 1
        if frame in motion_valid: count += 1
        if frame in size_valid: count += 1
        frame_validation_count[frame] = count
    
    final_valid = [frame for frame, count in frame_validation_count.items() if count >= 2]
    final_invalid = list(set(frames_data.keys()) - set(final_valid))
    
    print(f"최종 통과 프레임 수: {len(final_valid)}")
    
    return sorted(final_valid), sorted(final_invalid)


if __name__ == "__main__":
    videos_dir = r"D:\Project FIles\Ctrl_F\video"
    base_dir = "yolo_dataset"
    
    if not os.path.exists(videos_dir):
        print("비디오 폴더가 존재하지 않습니다!")
        exit()
    
    while True:
        # 프레임 추출 및 데이터셋 생성
        target_frames = 300
        process_videos(videos_dir, base_dir, target_frames)
        
        cont = input("\n계속해서 다른 비디오를 처리하시겠습니까? (y/n): ").lower()
        if cont != 'y':
            break
    

