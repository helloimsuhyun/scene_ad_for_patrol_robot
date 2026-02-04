# vlm.py (Qwen2-VL skeleton)
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

# Qwen2-VL
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration  # for Qwen2-VL


@dataclass
class VLMResult:
    text: str
    score: float
    extra: Dict[str, Any]


class VLMWrapper:

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.model_id = None

    def load(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        **kwargs,
    ):

        self.model_id = model_id

        model_kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
        return self

    def _to_pil(self, rgb: np.ndarray) -> Image.Image:
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        return Image.fromarray(rgb, mode="RGB")

    #vlm qwen chat temple (img, promt(질문))
    def _build_messages(self, image: Image.Image, q_image: Image.Image, prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    #gen된 promt에서 입력 promt부분은 제외하고 답변만 남긴 후 문자열로 decode
    def _decode_assistant(self, generated_ids: torch.Tensor, input_ids: torch.Tensor) -> str:
        gen = generated_ids[0]
        inp = input_ids[0]
        out_ids = gen[len(inp):]
        text = self.processor.tokenizer.decode(out_ids, skip_special_tokens=True)
        return text.strip()

    def _score_from_first_token_logits(
        self,
        first_step_logits: torch.Tensor,
        label_texts: Sequence[str],
    ) -> Dict[str, float]:
        """
        첫 생성 토큰의 logits으로 label 후보들의 확률을 대략 추정.
        - label_texts는 "OPEN", "CLOSED" 같은 단일 토큰/짧은 토큰열 권장
        - 구현은 스켈레톤: 각 label을 tokenize 했을 때 첫 토큰 기준으로 점수화
          (정교하게 하려면 label이 여러 토큰일 때 길이 보정/누적 logprob 필요)
        """
        tok = self.processor.tokenizer
        probs = torch.softmax(first_step_logits.float(), dim=-1)  # (vocab,)

        out: Dict[str, float] = {}
        for lab in label_texts:
            ids = tok(lab, add_special_tokens=False).input_ids
            if len(ids) == 0:
                out[lab] = 0.0
                continue
            out[lab] = float(probs[ids[0]].item())
        return out

    @torch.no_grad()
    def infer_one(
        self,
        pair_rgb: np.ndarray,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 0.0,
        label_texts: Optional[Sequence[str]] = None,
    ) -> VLMResult:

        assert self.model is not None and self.processor is not None

        image = self._to_pil(pair_rgb)

        # Qwen2-VL chat message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.0 if not do_sample else float(temperature),
        )

        want_score = label_texts is not None and len(label_texts) > 0

        if want_score:
            gen_out = self.model.generate(
                **inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = gen_out.sequences
            first_logits = gen_out.scores[0][0]
        else:
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            first_logits = None

        text_out = self._decode_assistant(generated_ids, inputs["input_ids"])

        score = 0.0
        extra = {}

        if want_score and first_logits is not None:
            label_probs = self._score_from_first_token_logits(first_logits, label_texts)
            best_lab = max(label_probs, key=label_probs.get)
            score = float(label_probs[best_lab])
            extra["label_probs_first_token"] = label_probs
            extra["best_label"] = best_lab

        return VLMResult(text=text_out, score=score, extra=extra)


    def infer_candidates(
        self,
        candidates: List[Dict[str, Any]],
        prompt: str,
        topk: Optional[int] = None,
        *,
        max_new_tokens: int = 16,
        label_texts: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        candidates: DINO_change_map.prepare_vlm_pairs_from_rois 결과
        return: candidate dict에 vlm 결과를 붙여서 반환
        """
        if topk is not None:
            candidates = candidates[:topk]

        out = []
        for c in candidates:
            res = self.infer_one(
                c["pair_rgb"],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                label_texts=label_texts,
            )
            c2 = dict(c)
            c2["vlm_text"] = res.text
            c2["vlm_score"] = float(res.score)
            c2["vlm_extra"] = res.extra
            out.append(c2)

        # 점수 기준 정렬
        out.sort(key=lambda d: d.get("vlm_score", 0.0), reverse=True)
        return out
    
    