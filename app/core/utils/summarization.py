from typing import List
from ..models_loader import sum_tokenizer, sum_model
from ..config import DEVICE


def summarize_texts(texts: List[str]) -> str:
    joined = " [SEP] ".join(texts)
    prompt = "summarize: " + joined
    inputs = sum_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    summary_ids = sum_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )
    return sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)