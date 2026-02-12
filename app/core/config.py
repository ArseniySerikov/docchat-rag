from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str

    chroma_dir: str = "./chroma_db"
    collection_name: str = "docchat"
    chunk_size: int = 900
    chunk_overlap: int = 150
    
    top_k: int = 5
    use_mmr: bool = True
    fetch_k: int = 20
    score_threshold: float = 0.35

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_context_chars: int = 12000


settings = Settings()
