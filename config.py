import os
from dotenv import load_dotenv

load_dotenv()

llm_config = {
    "config_list": [{
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY")
    }],
    "temperature": 0.1,
    "timeout": 120,
    "cache_seed": 42
}