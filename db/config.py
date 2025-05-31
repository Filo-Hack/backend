import os
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict



class Config(BaseSettings):
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    DOCS_AMVERA_PATH: str = os.path.join(BASE_DIR, "amvera_data", "docs_amvera")
    PARSED_JSON_PATH: str = os.path.join(BASE_DIR, "amvera_data", "parsed_json")
    AMVERA_CHROMA_PATH: str = os.path.join(BASE_DIR, "amvera_data", "chroma_db")
    AMVERA_COLLECTION_NAME: str = "amvera_docs"
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    LM_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    YANDX_MODEL_NAME: str = "yandex-chat"
    # OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"
    # OPENAI_API_KEY: SecretStr
    LM_MODEL_NAME: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/.env")
settings = Config()