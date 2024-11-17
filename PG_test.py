import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
import time

class WebcamObjectDetector:
    def __init__(self):
        self.cap = None  # 초기화
        print("Florence 모델 초기화 중...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True
        )
        
        print(f"모델 초기화 완료 (Device: {self.device})")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다")

    def detect_objects(self, image, query):
        # Florence 모델용 입력 준비
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = task_prompt + query
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # 객체 검출 실행
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1
            )

        # 결과 처리
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        result = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        return result.get('<CAPTION_TO_PHRASE_GROUNDING>', {})

    def draw_results(self, frame, results):
        if not results or 'bboxes' not in results:
            return frame

        for box in results['bboxes']:
            x1, y1, x2, y2 = map(int, box)
            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 신뢰도 표시 (있는 경우)
            if 'scores' in results:
                confidence = results['scores'][0]
                cv2.putText(frame, f"{confidence:.2f}", 
                          (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9, (0, 255, 0), 2)

        return frame

    def run(self):
        print("\n찾고자 하는 물체를 입력하세요:")
        target_object = input().strip()
        
        print("\n조작 방법:")
        print("- SPACE: 현재 프레임 분석")
        print("- Q: 종료")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 프레임 미러링 (더 자연스러운 화면을 위해)
            frame = cv2.flip(frame, 1)
            
            # 현재 프레임 표시
            cv2.imshow('Webcam', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                print("\n이미지 분석 중...")
                
                # OpenCV BGR -> RGB 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # 객체 검출
                start_time = time.time()
                results = self.detect_objects(pil_image, target_object)
                process_time = time.time() - start_time
                
                # 결과 표시
                if results and 'bboxes' in results:
                    frame = self.draw_results(frame, results)
                    print(f"객체 감지 완료 (처리 시간: {process_time:.2f}초)")
                else:
                    print("객체를 찾을 수 없습니다.")
                
                # 결과 화면 표시
                cv2.imshow('Detection Result', frame)
                cv2.waitKey(0)
                cv2.destroyWindow('Detection Result')

        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    detector = WebcamObjectDetector()
    detector.run()