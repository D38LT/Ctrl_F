import gradio as gr
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import random
import numpy as np
import copy
import torch
from tkinter import Tk, Button, Entry, Label
import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

# 불필요한 flash_attn 요구 사항 수정
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

model_path = "microsoft/Florence-2-base"

# 수정된 import로 모델 로드
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="sdpa", trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def run_caption_to_phrase_grounding(image, text_input):
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = task_prompt + text_input

    # 입력 준비 및 GPU로 이동
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
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
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    print(parsed_answer)  # 모델의 출력 형식 확인 (디버깅용)
    
    if '<CAPTION_TO_PHRASE_GROUNDING>' in parsed_answer:
        return parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']
    else:
        print("Warning: 'bboxes' not found in result")
        return None

def draw_bounding_boxes(image, result):
    draw = ImageDraw.Draw(image)
    
    if result and "bboxes" in result and "labels" in result:
        for bbox, label in zip(result["bboxes"], result["labels"]):
            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, y0), label, fill="white")
    else:
        print("Error: 'bboxes' or 'labels' not found in result.")

def start_caption_to_phrase_grounding():
    text_input = entry.get()
    ret, frame = webcam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)

    result = run_caption_to_phrase_grounding(pil_image, text_input)

    if result:
        draw_bounding_boxes(pil_image, result)

    pil_image.show()

# Tkinter UI 설정
root = Tk()
root.title("Caption to Phrase Grounding")

label = Label(root, text="Enter text:")
label.pack()

entry = Entry(root)
entry.pack()

start_button = Button(root, text="Start", command=start_caption_to_phrase_grounding)
start_button.pack()

# 카메라 설정
webcam = cv2.VideoCapture(0)

root.mainloop()
