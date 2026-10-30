import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
c = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"), max_retries=0, timeout=20)
try:
    r = c.chat.completions.create(model=os.getenv("LLM_MODEL"), messages=[{"role": "user", "content": "hi"}])
    print("OK", r.choices[0].message.content)
except Exception as e:
    print(type(e).__name__, getattr(e, "status_code", None), getattr(e, "body", None) or e)