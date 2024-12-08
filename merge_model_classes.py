from ultralytics import YOLO
import torch
import os
import yaml

def merge_model_classes(model_path, output_path=None):
    """YOLO 모델의 중복 클래스 병합"""
    # 모델 로드
    model = YOLO(model_path)
    
    # 현재 클래스 정보 확인
    current_classes = dict(model.names)
    print("\n=== 현재 모델의 클래스 ===")
    for idx, name in current_classes.items():
        print(f"{idx}: {name}")
    
    # 중복 클래스 찾기
    class_mapping = {}
    for idx, name in list(current_classes.items()):
        if name in class_mapping:
            class_mapping[name].append(idx)
        else:
            class_mapping[name] = [idx]
    
    duplicates = {name: indices for name, indices in class_mapping.items() if len(indices) > 1}
    
    if not duplicates:
        print("\n중복된 클래스가 없습니다.")
        return
    
    print("\n=== 중복된 클래스 발견 ===")
    for name, indices in duplicates.items():
        print(f"클래스 '{name}': 인덱스 {indices}")
    
    # 새로운 클래스 매핑 생성
    new_classes = {}
    old_to_new_mapping = {}
    next_idx = 0
    
    for name, indices in class_mapping.items():
        new_classes[next_idx] = name
        for old_idx in indices:
            old_to_new_mapping[old_idx] = next_idx
        next_idx += 1
    
    print("\n모델 가중치 수정 중...")
    
    # 새 모델 생성
    new_model = YOLO(model_path)
    new_model.model.nc = len(new_classes)
    new_model.model.names = new_classes
    
    # 가중치 복사 및 병합
    with torch.no_grad():
        # 백본과 넥 레이어의 가중치 복사
        for i in range(len(new_model.model.model) - 1):  # 마지막 레이어(헤드) 제외
            old_layer = model.model.model[i]
            new_layer = new_model.model.model[i]
            
            # Conv 레이어 복사
            if hasattr(old_layer, 'conv'):
                new_layer.conv.weight.copy_(old_layer.conv.weight)
                if hasattr(old_layer.conv, 'bias') and old_layer.conv.bias is not None:
                    new_layer.conv.bias.copy_(old_layer.conv.bias)
            
            # BatchNorm 레이어 복사
            if hasattr(old_layer, 'bn'):
                new_layer.bn.weight.copy_(old_layer.bn.weight)
                new_layer.bn.bias.copy_(old_layer.bn.bias)
                new_layer.bn.running_mean.copy_(old_layer.bn.running_mean)
                new_layer.bn.running_var.copy_(old_layer.bn.running_var)
        
        # 헤드 레이어 가중치 병합
        old_head = model.model.model[-1]
        new_head = new_model.model.model[-1]
        
        # Box regression 가중치 복사
        if hasattr(old_head, 'reg'):
            new_head.reg.weight.copy_(old_head.reg.weight)
            if hasattr(old_head.reg, 'bias') and old_head.reg.bias is not None:
                new_head.reg.bias.copy_(old_head.reg.bias)
        
        # Classification 가중치 병합
        if hasattr(old_head, 'cls'):
            old_cls_weight = old_head.cls.weight
            old_cls_bias = old_head.cls.bias
            new_cls_weight = new_head.cls.weight
            new_cls_bias = new_head.cls.bias
            
            for old_idx, new_idx in old_to_new_mapping.items():
                if old_idx < old_cls_weight.shape[0]:
                    new_cls_weight[new_idx] = old_cls_weight[old_idx]
                    if old_cls_bias is not None:
                        new_cls_bias[new_idx] = old_cls_bias[old_idx]
    
    # 모델 저장 경로 설정
    if output_path is None:
        base_dir = os.path.dirname(model_path)
        filename = os.path.basename(model_path)
        output_path = os.path.join(base_dir, f"merged_{filename}")
        print(f"\n저장 경로가 지정되지 않아 다음 경로에 저장됩니다: {output_path}")
    
    # 모델과 설정 저장
    new_model.save(output_path)
    
    # YAML 파일 생성
    yaml_path = os.path.splitext(output_path)[0] + '.yaml'
    yaml_content = {
        'names': [name for _, name in sorted(new_classes.items())],
        'nc': len(new_classes)
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True)
    
    print(f"\n병합된 모델이 저장됨: {output_path}")
    print(f"클래스 정보가 저장됨: {yaml_path}")
    print("\n=== 병합된 모델의 클래스 ===")
    for idx, name in sorted(new_classes.items()):
        print(f"{idx}: {name}")

if __name__ == "__main__":
    # 기본 모델 경로 설정
    default_model_path = r"D:\Project FIles\Ctrl_F\florence-2-large-playground-master\runs\detect\yolo_ctrlf10\weights\best.pt"
    
    if not os.path.exists(default_model_path):
        model_path = input("모델 파일을 찾을 수 없습니다. 다른 경로를 입력하세요: ").strip()
    else:
        print(f"기본 모델 경로: {default_model_path}")
        model_path = default_model_path
    
    # 출력 경로 설정 (선택사항)
    output_path = input("저장할 모델의 경로를 입력하세요 (Enter 시 자동 생성): ").strip()
    if not output_path:
        output_path = None
    
    try:
        merge_model_classes(model_path, output_path)
        print("\n모델 병합이 완료되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        # 디버깅을 위한 모델 구조 출력
        print("\n=== 모델 구조 ===")
        model = YOLO(model_path)
        for name, module in model.model.named_modules():
            print(f"{name}: {type(module)}") 