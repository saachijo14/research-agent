import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
c = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"),
           timeout=30, max_retries=0)

for model in [os.getenv("LLM_MODEL"), "gemini-3.5-flash-lite", "gemini-3.5-flash"]:
    t = time.time()
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Reply with JSON {"ok": true}'}],
            response_format={"type": "json_object"},
        )
        print("OK  ", model, round(time.time() - t, 1), "s", r.choices[0].message.content)
    except Exception as e:
        print("FAIL", model, type(e).__name__, getattr(e, "status_code", None),
              getattr(e, "body", None) or e)