import cv2
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import numpy as np
import time
import os
import re
# Flash attention 제거 수정
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

# 모델과 프로세서 로드
model_path = "microsoft/Florence-2-base"
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="sdpa", trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# GPU로 모델 이동
if torch.cuda.is_available():
    model.to("cuda")

def parse_generated_text(generated_text):
    # 객체 이름 추출
    label = re.search(r"</s><s>(.*?)<loc_", generated_text)
    label = label.group(1) if label else "Unknown"

    # 위치 정보 추출
    loc_matches = re.findall(r"<loc_(\d+)>", generated_text)
    if len(loc_matches) >= 4:
        # 필요한 4개의 좌표를 float로 변환하여 사용
        x0, y0, x1, y1 = [float(loc) for loc in loc_matches[:4]]
        bboxes = [[x0, y0, x1, y1]]
        labels = [label]
        return {"bboxes": bboxes, "labels": labels}
    else:
        print("Warning: Not enough location data found.")
        return None





# 객체 감지를 위한 함수
def run_caption_to_phrase_grounding(image, text_input):
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = task_prompt + text_input

    # 입력 준비 및 GPU로 이동
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    print("Prepared inputs:", inputs)  # 입력 확인
    
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.cuda().long() if k == "input_ids" else v.cuda().float() for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    if torch.cuda.is_available():
        model.to("cpu")
        torch.cuda.empty_cache()
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print("Generated text:", generated_text)  # 모델의 출력 텍스트 확인

    parsed_answer = parse_generated_text(generated_text)
    print("Parsed answer:", parsed_answer)  # 최종 결과 확인
    
    if 'bboxes' not in parsed_answer:
        print("Warning: 'bboxes' not found in result")
        return None

    return parsed_answer


# 바운딩 박스 그리기
def draw_bounding_boxes(image, result):
    draw = ImageDraw.Draw(image)
    
    if result and "bboxes" in result and "labels" in result:
        for bbox, label in zip(result["bboxes"], result["labels"]):
            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, y0), label, fill="white")
    else:
        print("Error: 'bboxes' or 'labels' not found in result.")
    return image

# 실시간 스트리밍
def main_loop(text_input, delay=1.0):
    cap = cv2.VideoCapture(0)  # 웹캠 연결
    last_inference_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 PIL 이미지로 변환 후 모델에 전달
        
        
        # 일정 시간마다 모델에 프레임 전달하여 바운딩 박스 업데이트
        current_time = time.time()
        if current_time - last_inference_time > delay:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = run_caption_to_phrase_grounding(pil_image, text_input)
            last_inference_time = current_time
            if result:
                pil_image = draw_bounding_boxes(pil_image, result)
        
        # 바운딩 박스가 그려진 이미지를 얻음
        

        # 이미지를 다시 OpenCV 형식으로 변환하여 화면에 표시
        cv2.imshow("Real-Time Object Detection", cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # GPU 메모리 관리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cap.release()
    cv2.destroyAllWindows()

# 사용자 입력
text_input = input("Enter the object description for grounding: ")
main_loop(text_input, delay=1.0)  # 프레임 처리 간격 설정 (1초 간격으로 모델 실행)
