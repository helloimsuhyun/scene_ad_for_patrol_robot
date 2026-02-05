# vlm.py (single-image VQA only)
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


class VLMWrapper:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load(self, model_id="Qwen/Qwen2-VL-2B-Instruct", **kwargs):
        # 4-bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # 또는 torch.float16
        )

        print(f"Loading model {model_id} in 4-bit quantization...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto", # 자동으로 GPU에 분산 배치
            **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        return self

    def _to_pil(self, rgb: np.ndarray) -> Image.Image:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    def _build_prompt(self, question: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def infer_image(self, img_rgb: np.ndarray, question: str, max_new_tokens: int = 16) -> str:
        assert self.model is not None and self.processor is not None

        image = self._to_pil(img_rgb)
        prompt = self._build_prompt(question)

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # 입력 프롬프트 제거 답변만 디코드
        gen_ids = out[0][len(inputs["input_ids"][0]):]
        answer = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return answer.strip()
