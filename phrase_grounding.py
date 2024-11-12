import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import psutil

def select_image():
    """파일 선택 대화상자"""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title='이미지를 선택하세요',
        filetypes=[('이미지 파일', '*.jpg *.jpeg *.png *.bmp'), ('모든 파일', '*.*')]
    )

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

def print_system_status():
    print("\n=== 시스템 상태 ===")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리 할당: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    process = psutil.Process(os.getpid())
    print(f"CPU 메모리 사용: {process.memory_info().rss / 1024**2:.2f} MB")

class FlorenceGrounding:
    def __init__(self):
        # Florence 모델과 프로세서 초기화
        model_name = "microsoft/Florence-2-base"
        print(f"모델 로드 중: {model_name}")
        
        # 모델 로드 전
        print("모델 로드 전 상태")
        print_system_status()
        
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                attn_implementation="sdpa", 
                trust_remote_code=True
            )
        
        # 모델 로드 후
        print("\n모델 로드 후 상태")
        print_system_status()
        
        # GPU 사용 가능시 GPU 사용
        if torch.cuda.is_available():
            self.model.to("cuda")
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"모델 로드 완료 (Device: {'cuda' if torch.cuda.is_available() else 'cpu'})")

    def resize_image_for_display(self, image, max_width=1280, max_height=720):
        """디스플레이를 위한 이미지 크기 조절"""
        height, width = image.shape[:2]
        
        if width <= max_width and height <= max_height:
            return image
        
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def process_image(self, image_path, text_queries, confidence_threshold=0.5):
        """이미지에서 객체 탐지"""
        # 이미지 로드
        image = Image.open(image_path)
        cv_image = cv2.imread(image_path)
        
        display_image = cv_image.copy()
        detected_objects = []
        
        for query in text_queries:
            # Phrase Grounding 실행
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            prompt = task_prompt + query
            
            inputs = self.processor(
                text=prompt,
                images=image, 
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                )
            
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
            result = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height)
            )
            
            if '<CAPTION_TO_PHRASE_GROUNDING>' in result:
                grounding_result = result['<CAPTION_TO_PHRASE_GROUNDING>']
                if grounding_result and "bboxes" in grounding_result:
                    confidence = grounding_result.get("scores", [1.0])[0]
                    if confidence >= confidence_threshold:
                        box = grounding_result["bboxes"][0]
                        detected_objects.append({
                            'label': query,
                            'confidence': confidence,
                            'box': box
                        })
                        
                        # 박스 그리기
                        cv2.rectangle(display_image, 
                                    (int(box[0]), int(box[1])), 
                                    (int(box[2]), int(box[3])), 
                                    (0, 255, 0), 2)
        
        return display_image, detected_objects

    def run(self):
        """메인 실행 함수"""
        # 이미지 선택
        image_path = select_image()
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
        display_image, detected_objects = self.process_image(
            image_path, 
            text_queries,
            confidence_threshold=0.5
        )
        
        # 결과 출력
        print("\n탐지된 객체:")
        for obj in detected_objects:
            print(f"- {obj['label']}: {obj['confidence']:.2f}")
        
        # 결과 저장
        output_dir = "grounding_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_grounded.jpg")
        cv2.imwrite(output_path, display_image)
        print(f"\n결과 이미지 저장됨: {output_path}")
        
        # 결과 표시
        display_image = self.resize_image_for_display(display_image)
        window_name = f"Grounding Result - {Path(image_path).name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_image)
        
        # 창 크기 자동 조절
        height, width = display_image.shape[:2]
        cv2.resizeWindow(window_name, width, height)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    grounding = FlorenceGrounding()
    grounding.run()