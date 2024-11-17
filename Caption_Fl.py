import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import numpy as np

# Florence 모델 로드
model_path = "microsoft/Florence-2-base"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

if torch.cuda.is_available():
    model.to("cuda")
    print("GPU 사용 중")

def detect_objects(image):
    """이미지에서 사람이 들고 있는 물체 감지"""
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>a person holding"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
            do_sample=False,
            num_beams=3,
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_result = processor.post_process_generation(
        generated_text,
        task="<CAPTION_TO_PHRASE_GROUNDING>",
        image_size=(image.width, image.height)
    )
    
    return parsed_result.get('<CAPTION_TO_PHRASE_GROUNDING>', None)

def get_object_name(image, bbox):
    """감지된 영역의 물체 이름 확인"""
    # 바운딩 박스 영역 추출
    crop_box = (
        max(0, int(bbox[0])),
        max(0, int(bbox[1])),
        min(image.width, int(bbox[2])),
        min(image.height, int(bbox[3]))
    )
    cropped_image = image.crop(crop_box)
    
    task_prompt = "<CAPTION>"
    inputs = processor(text=task_prompt, images=cropped_image, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=20,
            do_sample=False,
            num_beams=3,
        )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption.strip()

def main():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    process_every_n_frames = 30  # 10프레임마다 처리
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            # 처리하지 않는 프레임은 박스만 그리고 건너뛰기
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        # 이미지 크기 축소 (성능 향상)
        frame = cv2.resize(frame, (640, 480))
        
        # OpenCV BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # 물체 감지
        result = detect_objects(pil_image)
        
        if result and "bboxes" in result and result["bboxes"]:
            for bbox in result["bboxes"]:
                object_name = get_object_name(pil_image, bbox)
                
                print(f"물체 이름: {object_name}")
                # 박스 그리기
                start_point = (int(bbox[0]), int(bbox[1]))
                end_point = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(frame, object_name, 
                          (start_point[0], start_point[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()