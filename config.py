from dotenv import load_dotenv
import os


load_dotenv()

'''
全局配置
'''

class Config:


    OPENAI_API_KEY = os.getenv(
        "OPENAI_API_KEY"
    )


    OPENAI_BASE_URL = os.getenv(
        "OPENAI_BASE_URL"
    )


    LLM_MODEL = os.getenv(
        "OPENAI_MODEL_NAME",
        "DeepSeek-V4-Flash-0731"
    )

