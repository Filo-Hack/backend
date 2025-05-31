from ..models_loader import rec_tokenizer, rec_model
from ..config import DEVICE


def generate_recommendation(profile_summary: str, context: str) -> str:
    prompt = (
        f"Профиль пользователя: {profile_summary}\n"
        f"Контекст: {context}\n"
        "Задача: сформулируй короткую дружелюбную рекомендацию "
        "или вопрос о том, запускать ли привычную рутину. Ответ дай на русском.\n"
        "Ответ:"
    )
    inputs = rec_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    out_ids = rec_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )
    return rec_tokenizer.decode(out_ids[0], skip_special_tokens=True)