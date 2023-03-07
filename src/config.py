"""Config file for FastAPI"""
from pydantic import BaseSettings

metadata = [
    {
        "name": "segformer",
        "description": "SegFormer"
    }
]

class Settings(BaseSettings):
    """This class is the settings for the api"""
    title: str = "SegFormer API"
    description: str = "List of APIs for SegFormer"
    version: str = "0.0.1"
    openapi_tags: list = metadata
    
