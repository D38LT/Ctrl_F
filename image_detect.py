import cv2
from ultralytics import YOLO
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def select_image_path():
    """파일 선택 대화상자를 통해 이미지 또는 폴더 선택"""
    root = tk.Tk()
    root.withdraw()  # tkinter 창 숨기기
    
    # 파일/폴더 선택 옵션 제공
    choice = tk.messagebox.askquestion(
        "선택", 
        "여러 이미지를 처리하시겠습니까?\n\n'예' - 폴더 선택\n'아니오' - 단일 이미지 선택"
    )
    
    if choice == 'yes':
        # 폴더 선택
        path = filedialog.askdirectory(
            title='처리할 이미지가 있는 폴더를 선택하세요'
        )
    else:
        # 단일 이미지 파일 선택
        path = filedialog.askopenfilename(
            title='처리할 이미지를 선택하세요',
            filetypes=[
                ('이미지 파일', '*.jpg *.jpeg *.png *.bmp'),
                ('모든 파일', '*.*')
            ]
        )
    
    return path

def resize_image_for_display(image, max_width=1280, max_height=720):
    """디스플레이를 위한 이미지 크기 조절"""
    height, width = image.shape[:2]
    
    # 이미지가 최대 크기보다 작으면 그대로 반환
    if width <= max_width and height <= max_height:
        return image
    
    # 가로세로 비율 유지하면서 크기 조절
    ratio = min(max_width / width, max_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def detect_objects_in_image(image_path, conf_threshold=0.25, save_results=True):
    """이미지에서 객체 탐지"""
    # 모델 경로 고정
    model_path = r"D:\Project FIles\Ctrl_F\florence-2-large-playground-master\runs\detect\yolo_ctrlf4\weights\best.pt"
    
    # 모델 로드
    model = YOLO(model_path)
    print(f"모델 로드 완료: {model_path}")
    print(f"탐지 가능한 클래스: {model.names}")
    
    # 이미지 로드
    if not os.path.exists(image_path):
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return
    
    # 결과 저장 디렉토리 생성
    if save_results:
        output_dir = "detection_results"
        os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 읽을 수 없습니다!")
        return
    
    # 원본 이미지 보존
    display_image = image.copy()
    
    # 객체 탐지 실행
    results = model.predict(
        source=image,
        conf=conf_threshold,
        verbose=False
    )
    
    # 결과 처리
    detected_objects = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 클래스 이름과 신뢰도
            cls = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            
            # 결과 저장
            detected_objects.append({
                'class': cls,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })
            
            # 박스 그리기
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 텍스트 그리기
            label = f"{cls}: {conf:.2f}"
            # 텍스트 배경 추가
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(display_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # 결과 저장 (원본 크기)
    if save_results and detected_objects:
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_detected.jpg")
        cv2.imwrite(output_path, display_image)
        print(f"결과 이미지 저장됨: {output_path}")
    
    # 결과 출력
    print("\n탐지된 객체:")
    for obj in detected_objects:
        print(f"- {obj['class']}: {obj['confidence']:.2f}")
    
    # 디스플레이를 위한 이미지 크기 조절
    display_image = resize_image_for_display(display_image)
    
    # 창 이름 설정 (파일명 포함)
    window_name = f"Detection Result - {Path(image_path).name}"
    
    # 결과 이미지 표시
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_image)
    
    # 창 크기 자동 조절
    height, width = display_image.shape[:2]
    cv2.resizeWindow(window_name, width, height)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detected_objects

def process_multiple_images(image_dir, conf_threshold=0.25):
    """디렉토리 내의 모든 이미지 처리"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith(image_extensions):
            image_path = os.path.join(image_dir, file_name)
            print(f"\n처리 중: {file_name}")
            detect_objects_in_image(image_path, conf_threshold)

if __name__ == "__main__":
    # 파일/폴더 선택
    path = select_image_path()
    
    if not path:
        print("선택된 경로가 없습니다!")
        exit()
    
    # 선택된 경로에 따라 처리
    if os.path.isfile(path):
        detect_objects_in_image(path)
    elif os.path.isdir(path):
        process_multiple_images(path)
    else:
        print("유효하지 않은 경로입니다!")