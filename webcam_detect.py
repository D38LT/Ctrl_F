import cv2
import time
from ultralytics import YOLO
import numpy as np

def run_webcam_detection(conf_threshold=0.25):
    """웹캠에서 실시간으로 객체 탐지"""
    # 모델 경로 고정
    model_path = r"D:\Project FIles\Ctrl_F\florence-2-large-playground-master\runs\detect\yolo_ctrlf6\weights\best.pt"
    
    # 모델 로드
    model = YOLO(model_path)
    print(f"모델 로드 완료: {model_path}")
    print(f"탐지 가능한 클래스: {model.names}")
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다!")
        return
    
    # 웹캠 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS 계산을 위한 변수
    prev_time = time.time()
    fps = 0
    
    # 다중 스케일 설정
    scales = [0.8, 1.0, 1.2]
    
    print("'q'를 누르면 종료됩니다.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다!")
            break
        
        # FPS 계산
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        height, width = frame.shape[:2]
        all_predictions = []
        
        # 다중 스케일 추론
        for scale in scales:
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            resized = cv2.resize(frame, (new_width, new_height))
            results = model.predict(
                source=resized,
                conf=conf_threshold,
                iou=0.5,
                agnostic_nms=True,
                verbose=False
            )
            
            # 원본 크기로 좌표 변환
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0] / scale
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    all_predictions.append((x1, y1, x2, y2, cls, conf))
        
        # NMS로 중복 제거
        final_boxes = apply_nms(all_predictions, iou_threshold=0.5)
        
        # 결과 시각화
        for x1, y1, x2, y2, cls, conf in final_boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_name = model.names[cls]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # FPS 표시
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 결과 표시
        cv2.imshow("Object Detection", frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()

def apply_nms(predictions, iou_threshold=0.5):
    """비최대 억제(NMS) 적용"""
    if not predictions:
        return []
        
    # 신뢰도 기준 정렬
    predictions = sorted(predictions, key=lambda x: x[5], reverse=True)
    
    final_boxes = []
    while predictions:
        current = predictions.pop(0)
        predictions = [
            pred for pred in predictions
            if calculate_iou(current[:4], pred[:4]) < iou_threshold
            or current[4] != pred[4]  # 다른 클래스는 유지
        ]
        final_boxes.append(current)
    
    return final_boxes

def calculate_iou(box1, box2):
    """IoU(Intersection over Union) 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (box1_area + box2_area - intersection)

if __name__ == "__main__":
    run_webcam_detection(conf_threshold=0.25)