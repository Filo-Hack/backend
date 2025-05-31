from typing import List
from ..models_loader import emb_model

async def generate_embedding(text: str) -> List[float]:
    emb = emb_model.encode([text], convert_to_numpy=True)
    return emb[0].tolist()