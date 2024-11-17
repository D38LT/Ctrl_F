import cv2
import time
import numpy as np
from ultralytics import YOLO

class ModelComparator:
    def __init__(self, model_paths):
        self.models = [YOLO(path) for path in model_paths]
        self.model_names = [f"Model {i+1}" for i in range(len(model_paths))]
        self.colors = [(0,255,0), (0,0,255), (255,0,0)]  # BGR 색상
        
    def run_comparison(self, conf_threshold=0.25):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 성능 측정을 위한 시작 시간
            start_times = []
            inference_times = []
            frames = []
            
            # 각 모델별 추론 실행
            for model in self.models:
                start_time = time.time()
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                inference_time = time.time() - start_time
                
                start_times.append(start_time)
                inference_times.append(inference_time)
                
                # 결과 시각화
                frame_copy = frame.copy()
                for r in results[0].boxes:
                    box = r.xyxy[0].cpu().numpy()
                    conf = float(r.conf[0])
                    cls = int(r.cls[0])
                    
                    x1, y1, x2, y2 = map(int, box)
                    color = self.colors[self.models.index(model)]
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.putText(frame_copy, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # FPS 표시
                fps = 1.0 / inference_time
                cv2.putText(frame_copy, f"{self.model_names[self.models.index(model)]}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame_copy, f"FPS: {fps:.1f}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                frames.append(frame_copy)
            
            # 화면 분할 표시
            if len(frames) == 2:
                combined = np.hstack((frames[0], frames[1]))
            else:
                top_row = np.hstack((frames[0], frames[1]))
                bottom_row = np.hstack((frames[2], np.zeros_like(frames[2])))  # 빈 공간 추가
                combined = np.vstack((top_row, bottom_row))
            
            # 크기 조정
            scale = 0.8
            combined = cv2.resize(combined, None, fx=scale, fy=scale)
            
            cv2.imshow('Model Comparison', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def compare_metrics(self, data_yaml):
        """모델들의 성능 메트릭 비교"""
        results = []
        for i, model in enumerate(self.models):
            print(f"\n평가 중: {self.model_names[i]}")
            result = model.val(data=data_yaml)
            results.append({
                'model': self.model_names[i],
                'mAP50': result.box.map50,
                'mAP50-95': result.box.map,
                'precision': result.box.p,
                'recall': result.box.r
            })
        
        # 결과 출력
        for r in results:
            print(f"\n{r['model']} 성능:")
            print(f"mAP50: {r['mAP50']:.3f}")
            print(f"mAP50-95: {r['mAP50-95']:.3f}")
            print(f"Precision: {r['precision']:.3f}")
            print(f"Recall: {r['recall']:.3f}")

if __name__ == "__main__":
    model_paths = [
        r"D:\Project FIles\Ctrl_F\florence-2-large-playground-master\runs\detect\yolo_ctrlf\weights\best.pt",
        r"D:\Project FIles\Ctrl_F\florence-2-large-playground-master\runs\detect\yolo_ctrlf2\weights\best.pt",
        r"D:\Project FIles\Ctrl_F\florence-2-large-playground-master\runs\detect\yolo_ctrlf4\weights\best.pt"
    ]
    
    comparator = ModelComparator(model_paths)
    
    # 실시간 비교 실행
    comparator.run_comparison(conf_threshold=0.25)
    
    # 메트릭 비교 (선택사항)
    #comparator.compare_metrics("path/to/data.yaml") 