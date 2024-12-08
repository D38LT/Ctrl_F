import cv2
import time
from ultralytics import YOLO
import numpy as np

# 파일 상단에 전역 변수 추가
reference_boxes = {}  # 클래스별 권장 위치 저장
current_class_idx = 0  # 현재 설정 중인 클래스 인덱스
drawing = False
ix, iy = -1, -1
frame = None  # frame 변수 추가
model = None  # model 변수 추가
temp_frame = None  # 임시 프레임 저장용

class ScoreSystem:
    def __init__(self):
        self.score = 100
        self.mistake_counts = {}  # 클래스별 실수 횟수 기록
        self.active_objects = set()  # 현재 화면에 있는 객체들 추적
        self.last_penalty_time = {}  # 클래스별 마지막 감점 시간
        self.last_bonus_time = time.time()  # 마지막 보너스 점수 시간
        self.all_correct_start_time = None  # 모든 물체가 올바른 위치에 있기 시작한 시간
        self.bonus_timer_active = False  # 보너스 타이머 활성화 상태
        
    def update_score(self, detections, reference_boxes):
        current_time = time.time()
        all_correct = True
        active_classes = set()
        wrong_positions = False
        
        # 현재 탐지된 모든 물체 확인
        for cls, (box, conf) in detections.items():
            if conf < 0.5:  # confidence가 낮으면 무시
                continue
                
            active_classes.add(cls)
            
            # reference_boxes에 해당 클래스가 있는지 확인
            if cls in reference_boxes:
                is_correct = is_center_inside_reference(box.xyxy[0], reference_boxes[cls])
                
                if not is_correct:
                    all_correct = False
                    wrong_positions = True
                    # 10초마다 한 번씩만 감점
                    if current_time - self.last_penalty_time.get(cls, 0) >= 10:
                        self.score = max(0, self.score - 3)  # 기본 감점
                        self.mistake_counts[cls] = self.mistake_counts.get(cls, 0) + 1
                        
                        # 같은 물건 반복 실수 시 추가 감점
                        if self.mistake_counts[cls] > 1:
                            self.score = max(0, self.score - 2)
                        
                        self.last_penalty_time[cls] = current_time
                        print(f"감점 발생: {cls} 클래스 위치 오류")  # 디버깅용
        
        # 모든 물체가 올바른 위치에 있고 필요한 모든 클래스가 존재하는지 확인
        required_classes = set(reference_boxes.keys())
        if not active_classes.issuperset(required_classes):
            all_correct = False
            print("일부 필요한 클래스가 감지되지 않음")  # 디버깅용
        
        # 보너스 점수 및 타이머 관리
        if all_correct and active_classes:
            if self.all_correct_start_time is None:
                self.all_correct_start_time = current_time
                self.bonus_timer_active = True
                print("보너스 타이머 시작")  # 디버깅용
            elif self.bonus_timer_active and current_time - self.all_correct_start_time >= 60:
                self.score = min(100, self.score + 5)  # 5점 보너스
                self.last_bonus_time = current_time
                self.all_correct_start_time = current_time  # 타이머 리셋
                print("보너스 점수 획득!")  # 디버깅용
        else:
            if wrong_positions:
                self.all_correct_start_time = None
                self.bonus_timer_active = False
                print("잘못된 위치로 인한 타이머 리셋")  # 디버깅용
        
        self.active_objects = active_classes
        return self.score
    
    def show_score(self, frame):
        score_text = f"Score: {self.score}"
        cv2.putText(frame, score_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 보너스 타이머 표시
        if self.bonus_timer_active and self.all_correct_start_time is not None:
            time_left = 60 - (time.time() - self.all_correct_start_time)
            if time_left > 0:
                bonus_text = f"Bonus in: {int(time_left)}s"
                cv2.putText(frame, bonus_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 메인 탐지 루프에서 사용
score_system = ScoreSystem()  # 전역 변수로 추가

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, current_class_idx, reference_boxes, frame, model, temp_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        if frame is not None:
            temp_frame = frame.copy()
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and temp_frame is not None:
            img_copy = temp_frame.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            if current_class_idx < len(model.names):
                label = f"Setting : {model.names[current_class_idx]}"
                cv2.putText(img_copy, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Set Reference Positions", img_copy)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if frame is not None and model is not None and current_class_idx < len(model.names):
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            
            reference_boxes[current_class_idx] = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {current_class_idx}: {model.names[current_class_idx]}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"설정된 클래스: {model.names[current_class_idx]} ({current_class_idx})")
            current_class_idx += 1
            
            if current_class_idx < len(model.names):
                print(f"다음 클래스: {model.names[current_class_idx]} ({current_class_idx})")
            else:
                print("모든 클래스 설정 완료!")
            
            temp_frame = frame.copy()
            cv2.imshow("Set Reference Positions", frame)

def run_webcam_detection(conf_threshold=0.25):
    """웹캠에서 실시간으로 객체 탐지"""
    global frame, model, current_class_idx, temp_frame
    
    # 모델 경로 고정
    model_path = r"D:\Project FIles\Ctrl_F\florence-2-large-playground-master\runs\detect\yolo_ctrlf10\weights\merged_best.pt"
    
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
    
    # 권장 위치 설정 창 생성
    cv2.namedWindow("Set Reference Positions")
    cv2.setMouseCallback("Set Reference Positions", mouse_callback)
    
    print("\n=== 권장 위치 설정 모드 ===")
    print("각 클래스별 권장 위치를 설정하세요.")
    print("설정할 클래스들:", [f"{idx}: {name}" for idx, name in model.names.items()])
    print(f"\n현재 설정할 클래스: {model.names[current_class_idx]} ({current_class_idx})")
    
    # 권장 위치 설정 루프
    while current_class_idx < len(model.names):
        ret, frame = cap.read()
        if not ret:
            break
        
        if temp_frame is None:
            temp_frame = frame.copy()
        
        # 이미 설정된 박스들 표시
        display_frame = frame.copy()
        for cls_idx, box in reference_boxes.items():
            x1, y1, x2, y2 = box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cls_idx}: {model.names[cls_idx]}"
            cv2.putText(display_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 현재 설정 중인 클래스 표시
        cv2.putText(display_frame, f"Set Reference Position: {model.names[current_class_idx]}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Set Reference Positions", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow("Set Reference Positions")
    
    # 메인 탐지 루프 수정
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 먼저 권장 위치 박스들을 표시
        for cls_idx, box in reference_boxes.items():
            rx1, ry1, rx2, ry2 = box
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)  # 흰색 박스
            if cls_idx in model.names:  # 클래스 인덱스 확인
                ref_label = f"Target: {model.names[cls_idx]}"
                cv2.putText(frame, ref_label, (rx1, ry1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 탐지 결과 처리
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=0.5,
            agnostic_nms=True,
            verbose=False
        )
        
        # 클래스별 최고 confidence 객체 찾기
        best_detections = {}  # {class_id: (box, conf)}
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                cls = int(box.cls[0])
                if cls not in model.names:  # 유효하지 않은 클래스 인덱스 건너뛰기
                    continue
                    
                conf = float(box.conf[0])
                if cls not in best_detections or conf > best_detections[cls][1]:
                    best_detections[cls] = (box, conf)
        
        # 최고 confidence 객체만 처리
        for cls, (box, conf) in best_detections.items():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 권장 위치와 현재 위치 비교
            if cls in reference_boxes:
                is_correct_position = is_center_inside_reference(
                    [x1, y1, x2, y2], 
                    reference_boxes[cls]
                )
                color = (0, 255, 0) if is_correct_position else (0, 0, 255)
                
                # confidence score를 포함하 점수 업데이트
                score = score_system.update_score(best_detections, reference_boxes)
                score_system.show_score(frame)
            else:
                color = (0, 0, 255)
            
            # 중심점 표시
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
            
            # 박스와 레이블 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if cls in model.names:  # 클래스 인덱스 확인
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 점수 표시
        score_system.show_score(frame)
        
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

def is_center_inside_reference(detected_box, reference_box):
    """탐지된 객체의 중심점이 권장 위치 안에 있는지 확인"""
    # 탐지된 객체의 중심점 계산
    center_x = (detected_box[0] + detected_box[2]) / 2
    center_y = (detected_box[1] + detected_box[3]) / 2
    
    # 권장 위치 박스의 범위
    ref_x1, ref_y1, ref_x2, ref_y2 = reference_box
    
    # 중심점이 권장 위치 안에 있는지 확인
    return (ref_x1 <= center_x <= ref_x2) and (ref_y1 <= center_y <= ref_y2)

if __name__ == "__main__":
    run_webcam_detection(conf_threshold=0.25)