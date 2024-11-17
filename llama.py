import ollama
import cv2
import os
from datetime import datetime
import torch

def capture_and_analyze():
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Press SPACE to capture, Q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            temp_filename = f"temp_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(temp_filename, frame)
            
            print("사진 촬영 완료! GPU로 분석 중...")
            
            try:
                response = ollama.chat(
                    model='llama3.2-vision',
                    messages=[{
                        'role': 'user',
                        'content': 'What is the man holding?',
                        'images': [temp_filename]
                    }],
                    stream=False,
                    options={
                        "num_gpu": 1,
                        "num_thread": 8,
                        "gpu_layers":40
                    }
                )
                print(f"분석 결과: {response['message']['content']}")
            except Exception as e:
                print(f"분석 중 오류 발생: {e}")
            
            os.remove(temp_filename)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_analyze()