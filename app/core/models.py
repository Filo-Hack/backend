from typing import List, Optional, Dict
from pydantic import BaseModel


class AddDocRequest(BaseModel):
    doc_id: str
    document: str
    embedding: Optional[List[float]]
    metadata: Dict

class QueryRequest(BaseModel):
    query_embedding: List[float]
    n_results: Optional[int] = 5

class QueryResultItem(BaseModel):
    doc_id: str
    document: str
    distance: float
    metadata: Dict

class QueryResponse(BaseModel):
    results: List[QueryResultItem]

class SummarizeRequest(BaseModel):
    texts: List[str]

class SummarizeResponse(BaseModel):
    summary: str

class RecommendRequest(BaseModel):
    profile_summary: str
    context: str

class RecommendResponse(BaseModel):
    recommendation: str

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio_base64: str

class STTResponse(BaseModel):
    text: str

class ChatRequest(BaseModel):
    history: List[Dict]
    user_input: str

class ChatResponse(BaseModel):
    response: str
