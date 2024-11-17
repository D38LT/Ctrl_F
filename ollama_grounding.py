import cv2
import ollama
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

class OllamaGrounding:
    def __init__(self):
        self.model_name = 'llama3.2-vision'  # 또는 사용 가능한 vision 모델
        print(f"Ollama 초기화 완료 (모델: {self.model_name})")

    def select_image(self):
        root = tk.Tk()
        root.withdraw()
        return filedialog.askopenfilename(
            title='이미지를 선택하세요',
            filetypes=[('이미지 파일', '*.jpg *.jpeg *.png *.bmp'), ('모든 파일', '*.*')]
        )

    def process_image(self, image_path, text_queries):
        # 이미지를 바이트로 변환
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # 원본 이미지 로드
        cv_image = cv2.imread(image_path)
        display_image = cv_image.copy()
        detected_objects = []

        # Ollama에 쿼리 전송
        prompt = f"이 이미지에서 다음 물체들의 위치를 찾아주세요: {', '.join(text_queries)}. 각 물체에 대해 x1,y1,x2,y2 좌표로 응답해주세요."
        
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes]
            }]
        )

        # 응답 파싱 및 박스 그리기
        content = response['message']['content']
        height, width = cv_image.shape[:2]

        for query in text_queries:
            if query.lower() in content.lower():
                # 간단한 파싱 예시 (실제 응답 형식에 따라 수정 필요)
                try:
                    # 예시: 좌표 추출 로직
                    # 실제 Ollama의 응답 형식에 맞게 수정 필요
                    box = [width//4, height//4, width*3//4, height*3//4]  # 임시 박스
                    
                    detected_objects.append({
                        'label': query,
                        'box': box
                    })
                    
                    # 박스 그리기
                    cv2.rectangle(display_image, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                (0, 255, 0), 2)
                    
                    # 레이블 추가
                    cv2.putText(display_image, query, 
                              (int(box[0]), int(box[1]-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                              (0, 255, 0), 2)
                except:
                    print(f"'{query}' 객체의 좌표를 파싱할 수 없습니다.")

        return display_image, detected_objects, content

    def resize_image_for_display(self, image, max_width=1280, max_height=720):
        height, width = image.shape[:2]
        if width <= max_width and height <= max_height:
            return image
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def run(self):
        # 이미지 선택
        image_path = self.select_image()
        if not image_path:
            print("이미지가 선택되지 않았습니다.")
            return

        # 검색할 텍스트 쿼리 입력
        print("\n쉼표(,)로 구분하여 찾고자 하는 물체들을 입력하세요:")
        text_queries = [q.strip() for q in input().split(',')]

        if not text_queries:
            print("검색할 물체가 입력되지 않았습니다.")
            return

        # 이미지 처리
        display_image, detected_objects, raw_response = self.process_image(image_path, text_queries)

        # 결과 출력
        print("\n=== Ollama 응답 ===")
        print(raw_response)
        
        print("\n=== 탐지된 객체 ===")
        for obj in detected_objects:
            print(f"- {obj['label']}")

        # 결과 저장
        output_dir = "ollama_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_detected.jpg")
        cv2.imwrite(output_path, display_image)
        print(f"\n결과 이미지 저장됨: {output_path}")

        # 결과 표시
        display_image = self.resize_image_for_display(display_image)
        window_name = f"Ollama Detection - {Path(image_path).name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_image)
        
        height, width = display_image.shape[:2]
        cv2.resizeWindow(window_name, width, height)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    grounding = OllamaGrounding()
    grounding.run() 