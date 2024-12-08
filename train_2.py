from ultralytics import YOLO
import os
import torch
import yaml
import shutil

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

def get_available_classes(base_dir):
    """데이터셋에서 사용 가능한 클래스 목록 반환"""
    train_image_dir = os.path.join(base_dir, 'train', 'images')
    if not os.path.exists(train_image_dir):
        print("올바른 데이터셋 구조가 아닙니다!")
        return []
    
    return [d for d in os.listdir(train_image_dir) 
            if os.path.isdir(os.path.join(train_image_dir, d))]

def select_classes_for_training(available_classes, existing_classes=None):
    """학습할 클래스 선택"""
    print("\n=== 사용 가능한 클래스 ===")
    
    # 기존 클래스와 새로운 클래스를 모두 표시
    all_classes = []
    if existing_classes:
        print("\n기존 모델의 클래스:")
        for idx, name in existing_classes.items():
            print(f"{idx}. {name} (기존)")
            all_classes.append(name)
    
    # 새로운 클래스 표시
    new_classes = [c for c in available_classes if c not in all_classes]
    if new_classes:
        print("\n새로운 클래스:")
        start_idx = len(all_classes)
        for idx, class_name in enumerate(new_classes, start_idx):
            print(f"{idx}. {class_name} (새로운)")
    
    # 전체 클래스 목록 통합
    all_available_classes = all_classes + new_classes
    
    while True:
        try:
            selection = input("\n학습할 클래스 번호를 선택하세요 (쉼표로 구분, 전체는 'all'): ").strip()
            if selection.lower() == 'all':
                return all_available_classes
            
            indices = [int(idx.strip()) for idx in selection.split(',')]
            selected = [all_available_classes[idx] for idx in indices 
                       if 0 <= idx < len(all_available_classes)]
            
            if not selected:
                print("올바른 번호를 선택해주세요.")
                continue
            
            return selected
            
        except (ValueError, IndexError):
            print("올바른 형식으로 입력해주세요.")

def prepare_temp_dataset(base_dir, selected_classes, existing_classes=None):
    """선택된 클래스의 데이터를 임시 폴더에 준비"""
    temp_dir = "temp_dataset"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 평가할 전체 클래스 목록 생성
    all_classes = []
    if existing_classes:
        all_classes.extend(existing_classes.values())
    all_classes.extend(selected_classes)
    
    # 새로운 클래스 인덱스 매핑
    class_names = {}
    if existing_classes:
        class_names.update(existing_classes)
    start_idx = len(existing_classes) if existing_classes else 0
    for idx, cls in enumerate(selected_classes, start_idx):
        class_names[idx] = cls
    
    # 임시 데이터셋 구조 생성
    for split in ['train', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(temp_dir, split, subdir), exist_ok=True)
    
    # 모든 클래스의 데이터 복사
    for split in ['train', 'test']:
        for class_name in all_classes:
            src_img_dir = os.path.join(base_dir, split, 'images', class_name)
            if not os.path.exists(src_img_dir):
                continue
                
            for img in os.listdir(src_img_dir):
                if img.endswith(('.jpg', '.png')):
                    # 이미지 복사
                    shutil.copy2(
                        os.path.join(src_img_dir, img),
                        os.path.join(temp_dir, split, 'images', img)
                    )
                    
                    # 라벨 파일 수정 및 복사
                    label_file = os.path.splitext(img)[0] + '.txt'
                    src_label = os.path.join(base_dir, split, 'labels', class_name, label_file)
                    dst_label = os.path.join(temp_dir, split, 'labels', label_file)
                    
                    if os.path.exists(src_label):
                        with open(src_label, 'r') as f:
                            lines = f.readlines()
                        
                        # 클래스 ID 수정
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # 기존 클래스는 원래 인덱스 유지, �� 클래스는 새 인덱스 사용
                                class_idx = next(
                                    (k for k, v in class_names.items() if v == class_name),
                                    None
                                )
                                if class_idx is not None:
                                    parts[0] = str(class_idx)
                                    new_lines.append(' '.join(parts) + '\n')
                        
                        with open(dst_label, 'w') as f:
                            f.writelines(new_lines)
    
    # dataset.yaml 생성
    yaml_content = {
        'path': os.path.abspath(temp_dir),
        'train': 'train/images',
        'val': 'test/images',
        'test': 'test/images',
        'names': class_names,
        'nc': len(class_names)
    }
    
    yaml_path = os.path.join(temp_dir, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_content, f, allow_unicode=True, sort_keys=False)
    
    return temp_dir, yaml_path

def train_yolo(base_dir, weights_path=None, epochs=100):
    """YOLO 학습 함수"""
    available_classes = get_available_classes(base_dir)
    if not available_classes:
        print("학습할 수 있는 클래스가 없습니다!")
        return
    
    # 기��� 모델 클래스 정보 가져오기
    existing_classes = None
    if weights_path and os.path.exists(weights_path):
        existing_classes = get_model_classes(weights_path)
        print("\n=== 기존 모델의 클래스 정보 ===")
        for idx, name in existing_classes.items():
            print(f"- {idx}: {name}")
    
    # 학습할 클래스 선택 (기존 + 새로운 클래스 모두 선택 가능)
    selected_classes = select_classes_for_training(available_classes, existing_classes)
    if not selected_classes:
        return
    
    print("\n선택된 클래스:")
    for cls in selected_classes:
        status = "기존" if existing_classes and cls in existing_classes.values() else "새로운"
        print(f"- {cls} ({status})")
    
    # 임시 데이터셋 준비
    temp_dir, yaml_path = prepare_temp_dataset(base_dir, selected_classes, existing_classes)
    
    exp_dir = get_next_exp_dir()
    exp_name = os.path.basename(exp_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if weights_path and os.path.exists(weights_path):
        model = YOLO(weights_path)
    else:
        model = YOLO('yolov8n.pt')
    
    # 학습 파라미터 설정
    model.train(
        data=yaml_path,
        project='runs/detect',
        name=exp_name,
        epochs=epochs,
        exist_ok=True,
        imgsz=640,
        batch=16,
        workers=8,
        device=device,
        amp=True,
        
        # 데이터 증강
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
        
        # 최적화 파라미터
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # 조기 종료 및 모델 저장
        patience=20,
        
        # 추가 최적화
        cos_lr=True,
        label_smoothing=0.1,
    )
    
    # 학습 완료 후 모델 평가
    print("\n=== 모델 평가 시작 ===")
    model_dir = os.path.join('runs/detect', exp_name)
    best_weights = os.path.join(model_dir, 'weights', 'best.pt')
    
    # dataset.yaml 파일을 모델 폴더로 복사
    shutil.copy2(yaml_path, os.path.join(model_dir, 'dataset.yaml'))
    
    if os.path.exists(best_weights):
        evaluate_model(best_weights, yaml_path)
    else:
        print("최적 가중치 파일을 찾을 수 없습니다!")
    
    # 임시 폴더 정리 (dataset.yaml 파일은 이미 복사됨)
    shutil.rmtree(temp_dir)

def evaluate_model(weights_path, data_yaml):
    """모델 성능 평가"""
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    
    print("\n=== 모델 평가 결과 ===")
    maps = metrics.maps  # mAP for each class at different IoU thresholds
    results = metrics.box.mean_results()  # [P, R, mAP50, mAP50-95]
    
    print(f"mAP50: {results[2]:.4f}")
    print(f"mAP50-95: {results[3]:.4f}")
    
    # 클래스별 성능
    print("\n클래스별 성능:")
    for i in metrics.box.ap_class_index:
        results = metrics.box.class_result(i)
        class_name = model.names[i]
        print(f"{class_name}:")
        print(f"  Precision: {results[0]:.4f}")
        print(f"  Recall: {results[1]:.4f}")
        print(f"  AP50: {results[2]:.4f}")
        print(f"  AP50-95: {results[3]:.4f}")
    
    # 전체 평균 성능
    results = metrics.box.mean_results()
    print("\n전체 평균 성능:")
    print(f"Mean Precision: {results[0]:.4f}")
    print(f"Mean Recall: {results[1]:.4f}")
    print(f"Mean AP50: {results[2]:.4f}")
    print(f"Mean AP50-95: {results[3]:.4f}")

def merge_duplicate_classes(base_dir):
    """중복된 클래스 병합"""
    classes_file = os.path.join(base_dir, 'classes.txt')
    if not os.path.exists(classes_file):
        print("classes.txt 파일을 찾을 수 없습니다.")
        return
    
    # 클래스 목록 읽기
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 중복 클래스 찾기
    class_indices = {}
    duplicates = {}
    for idx, class_name in enumerate(classes):
        if class_name in class_indices:
            duplicates[class_name] = [class_indices[class_name], idx]
        else:
            class_indices[class_name] = idx
    
    if not duplicates:
        print("중복된 클래스가 없습니다.")
        return
    
    print("\n=== 중복된 클래스 발견 ===")
    for class_name, indices in duplicates.items():
        print(f"클래스 '{class_name}': 인덱스 {indices[0]}, {indices[1]}")
        
        # 데이터 병합
        for split in ['train', 'test']:
            # 이미지 병합
            src_img_dir = os.path.join(base_dir, split, 'images', class_name)
            dst_img_dir = os.path.join(base_dir, split, 'images')
            
            if os.path.exists(src_img_dir):
                for img in os.listdir(src_img_dir):
                    src_path = os.path.join(src_img_dir, img)
                    dst_path = os.path.join(dst_img_dir, img)
                    if not os.path.exists(dst_path):
                        shutil.move(src_path, dst_path)
            
            # 라벨 병합
            src_label_dir = os.path.join(base_dir, split, 'labels', class_name)
            dst_label_dir = os.path.join(base_dir, split, 'labels')
            
            if os.path.exists(src_label_dir):
                for label in os.listdir(src_label_dir):
                    src_path = os.path.join(src_label_dir, label)
                    dst_path = os.path.join(dst_label_dir, label)
                    
                    # 라벨 파일의 클래스 인덱스 수정
                    with open(src_path, 'r') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            if int(parts[0]) == indices[1]:  # 중복된 인덱스
                                parts[0] = str(indices[0])   # 원래 인덱스로 변경
                            new_lines.append(' '.join(parts) + '\n')
                    
                    # 기존 파일에 추가 또는 새로 생성
                    mode = 'a' if os.path.exists(dst_path) else 'w'
                    with open(dst_path, mode) as f:
                        f.writelines(new_lines)
    
    # classes.txt 파일 업데이트
    unique_classes = list(dict.fromkeys(classes))  # 중복 제거
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name in unique_classes:
            f.write(f"{class_name}\n")
    
    print("\n클래스 병합이 완료되었습니다.")
    print(f"총 {len(unique_classes)}개의 고유 클래스가 있습니다.")

if __name__ == "__main__":
    base_dir = "yolo_dataset"
    weights_path = 'runs/detect/yolo_ctrlf8/weights/merged_best.pt'
    
    # 중복 클래스 병합 옵션 추가
    merge_option = input("중복된 클래스를 병합하시겠습니까? (y/n): ").lower()
    if merge_option == 'y':
        merge_duplicate_classes(base_dir)
    
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
    
    # 마지막으로 학습된 모델 평가
    try:
        exp_folders = [d for d in os.listdir('runs/detect') 
                      if d.startswith('yolo_ctrlf') and d[len('yolo_ctrlf'):].isdigit()]
        
        if exp_folders:
            latest_exp = max(exp_folders, 
                           key=lambda x: int(x[len('yolo_ctrlf'):]))
            latest_weights = os.path.join('runs/detect', latest_exp, 'weights', 'best.pt')
            latest_yaml = os.path.join('runs/detect', latest_exp, 'dataset.yaml')
            
            if os.path.exists(latest_weights) and os.path.exists(latest_yaml):
                print(f"\n=== 최종 모델 평가 ({latest_exp}) ===")
                evaluate_model(latest_weights, latest_yaml)
            else:
                print("\n최종 모델 파일을 찾을 수 없습니다.")
        else:
            print("\n학습된 모델이 없습니다.")
            
    except Exception as e:
        print(f"\n모델 평가 중 오류 발생: {e}")
