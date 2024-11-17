from ultralytics import YOLO
import os
import torch
import yaml

# 파일 상단에 전역변수 추가
exp_num = 1
base_exp_dir = 'runs/detect/yolo_ctrlf'

def get_next_exp_dir():
    """다음 실험 폴더 경로 반환"""
    global exp_num
    while os.path.exists(f'{base_exp_dir}{exp_num}'):
        exp_num += 1
    next_dir = f'{base_exp_dir}{exp_num}'
    os.makedirs(next_dir, exist_ok=True)  # 폴더 생성
    return next_dir

def get_model_classes(weights_path):
    """기존 모델의 클래스 정보 가져오기"""
    model = YOLO(weights_path)
    return model.names  # {0: 'class1', 1: 'class2', ...} 형태로 반환

def create_merged_dataset_yaml(base_dir, exp_dir, weights_path=None):
    """기존 모델 클래스와 새로운 클래스를 통합한 YAML 생성"""
    # exp_dir을 매개변수로 받아서 사용
    yaml_path = os.path.join(exp_dir, 'dataset.yaml')
    
    # 새로운 데이터셋의 클래스 정보 가져오기
    classes_file = os.path.join(base_dir, 'classes.txt')
    new_classes = {}
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            new_classes = {line.strip(): i for i, line in enumerate(f) if line.strip()}
    
    if not new_classes:
        print("새로운 클래스 정보가 없습니다!")
        return None
    
    # 기존 모델 클래스 정보 가져���기
    existing_classes = {}
    if weights_path and os.path.exists(weights_path):
        existing_classes = get_model_classes(weights_path)
        print("\n기존 모델 클래스:")
        for idx, name in existing_classes.items():
            print(f"- {idx}: {name}")
    
    # 클래스 통합 및 인덱스 재할당
    merged_classes = dict(existing_classes)  # 기존 클래스 복사
    next_index = len(existing_classes)
    
    # 로운 클래스 추가 (기존에 없는 경우만)
    added_classes = []
    for class_name in new_classes.keys():
        if class_name not in existing_classes.values():
            merged_classes[next_index] = class_name
            added_classes.append((next_index, class_name))
            next_index += 1
    
    # YAML 파일 생성
    yaml_content = {
        'path': os.path.abspath(base_dir),
        'train': 'train/images',
        'val': 'test/images',
        'test': 'test/images',
        'names': merged_classes,
        'nc': len(merged_classes)
    }
    
    # YAML 파일 생성 (폴더 생성 코드 제거)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_content, f, allow_unicode=True, sort_keys=False)
    
    print("\n통합된 클래스 구성:")
    for idx, name in merged_classes.items():
        print(f"- {idx}: {name}")
    
    if added_classes:
        print("\n새로 추가된 클래스:")
        for idx, name in added_classes:
            print(f"- {idx}: {name}")
    
    return yaml_path

def fix_dataset_labels(base_dir):
    """라벨 파일의 클래스 ID를 수정"""
    # classes.txt 읽기
    classes_file = os.path.join(base_dir, 'classes.txt')
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    num_classes = len(classes)
    print(f"등록된 클래스 수: {num_classes}")
    
    # 라벨 파일 수정
    fixed_count = 0
    for split in ['train', 'test']:
        label_dir = os.path.join(base_dir, split, 'labels')
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                file_path = os.path.join(label_dir, label_file)
                modified = False
                new_lines = []
                
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # YOLO 포맷 확인
                            class_id = int(parts[0])
                            if class_id >= num_classes:
                                print(f"수정: {label_file} - 클래스 ID {class_id} → {class_id % num_classes}")
                                parts[0] = str(class_id % num_classes)
                                modified = True
                        new_lines.append(' '.join(parts))
                
                if modified:
                    with open(file_path, 'w') as f:
                        f.write('\n'.join(new_lines))
                    fixed_count += 1
    
    print(f"수정된 라벨 파일 수: {fixed_count}")

def train_yolo(base_dir, weights_path=None, epochs=100):
    # 데이터셋 라벨 수정
    fix_dataset_labels(base_dir)
    
    # 실험 폴더 하나만 생성
    exp_dir = get_next_exp_dir()
    
    # yaml 파일 생성 (exp_dir 전달)
    yaml_path = create_merged_dataset_yaml(base_dir, exp_dir, weights_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if weights_path and os.path.exists(weights_path):
        model = YOLO(weights_path)
    else:
        model = YOLO('yolov8n.pt')
    
    # 동적 배치 크기 설정
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory
        batch_size = min(64, max(16, int(available_memory / (1024**3) * 8)))
    else:
        batch_size = 16
    
    model.train(
        data=yaml_path,
        name=os.path.basename(exp_dir),  # 폴더명만 사용
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        workers=8,
        device=device,
        amp=True,
        
        # 향상된 데이터 증강
        augment=True,
        degrees=20.0,
        translate=0.2,
        scale=0.5,
        shear=10.0,
        perspective=0.0001,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.4,
        auto_augment='randaugment',
        
        # 최적화 파라미터
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        # 조기 종료 및 모델 저장
        patience=20,
        #save_period=5,
        
        # 추가 최적화
        cos_lr=True,  # 코사인 학습률 스케줄링
        label_smoothing=0.1,  # 라벨 스무딩
        overlap_mask=True,  # 마스크 오버랩 허용
    )
def evaluate_model(model_path, data_yaml, img_size=640):
    """
    모델을 평가하여 성능을 출력하는 함수
    :param model_path: 학습된 모델의 경로
    :param data_yaml: 데이터셋 YAML 파일 경로
    :param img_size: 이미지 크기
    """
    print("모델 평가를 시작합니다...")
    model = YOLO(model_path)  # 학습된 모델 로드
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 결과 출력
    print("\n=== 모델 평가 결과 ===")
    print(f"mAP@0.5: {results.box.map:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map50_95:.4f}")
    print(f"Precision: {results.box.precision:.4f}")
    print(f"Recall: {results.box.recall:.4f}")
    return results

if __name__ == "__main__":
    base_dir = "yolo_dataset"
    weights_path = 'runs/detect/yolo_ctrlf/weights/best.pt'
    
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"현재 디바이스: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    if os.path.exists(weights_path):
        print("기존 모델이 발견되었습니다.")
        use_existing = input("기존 모델을 기반으로 학습하시겠습니까? (y/n): ").lower() == 'y'
        if use_existing:
            train_yolo(base_dir, weights_path)
        else:
            train_yolo(base_dir)
    else:
        train_yolo(base_dir) 
