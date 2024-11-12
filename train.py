from ultralytics import YOLO
import os
import torch

def get_model_classes(weights_path):
    """기존 모델의 클래스 정보 가져오기"""
    model = YOLO(weights_path)
    return model.names  # {0: 'class1', 1: 'class2', ...} 형태로 반환

def create_merged_dataset_yaml(base_dir, weights_path=None):
    """기존 모델 클래스와 새로운 클래스를 통합한 YAML 생성"""
    # 새로운 데이터셋의 클래스 정보 가져오기
    classes_file = os.path.join(base_dir, 'classes.txt')
    new_classes = {}
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            new_classes = {line.strip(): i for i, line in enumerate(f) if line.strip()}
    
    if not new_classes:
        print("새로운 클래스 정보가 없습니다!")
        return None
    
    # 기존 모델 클래스 정보 가져오기
    existing_classes = {}
    if weights_path and os.path.exists(weights_path):
        existing_classes = get_model_classes(weights_path)
        print("\n기존 모델 클래스:")
        for idx, name in existing_classes.items():
            print(f"- {idx}: {name}")
    
    # 클래스 통합 및 인덱스 재할당
    merged_classes = dict(existing_classes)  # 기존 클래스 복사
    next_index = len(existing_classes)
    
    # 새로운 클래스 추가 (기존에 없는 경우만)
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
    
    yaml_path = os.path.join(base_dir, 'dataset.yaml')
    import yaml
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

def train_yolo(base_dir, weights_path=None, epochs=100):
    yaml_path = create_merged_dataset_yaml(base_dir, weights_path)
    
    if weights_path and os.path.exists(weights_path):
        model = YOLO(weights_path)
    else:
        model = YOLO('yolov8n.pt')
    
    # 동적 배치 크기 설정
    available_memory = torch.cuda.get_device_properties(0).total_memory
    batch_size = min(64, max(16, int(available_memory / (1024**3) * 8)))
    
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        workers=8,
        device=0,
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
        save_period=5,
        name='yolo_ctrlf',
        
        # 추가 최적화
        cos_lr=True,  # 코사인 학습률 스케줄링
        label_smoothing=0.1,  # 라벨 스무딩
        overlap_mask=True,  # 마스크 오버랩 허용
    )

if __name__ == "__main__":
    base_dir = "yolo_dataset"
    weights_path = 'runs/detect/yolo_ctrlf4/weights/best.pt'
    
    if os.path.exists(weights_path):
        print("기존 모델이 발견되었습니다.")
        use_existing = input("기존 모델을 기반으로 학습하시겠습니까? (y/n): ").lower() == 'y'
        if use_existing:
            train_yolo(base_dir, weights_path)
        else:
            train_yolo(base_dir)
    else:
        train_yolo(base_dir) 