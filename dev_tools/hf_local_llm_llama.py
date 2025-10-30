from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, traceback

# llama-cpp-python Llama wrapper
try:
    from llama_cpp import Llama
    _HAS_LLAMA = True
except Exception as e:
    print("llama-cpp-python not available:", e)
    _HAS_LLAMA = False

app = FastAPI()

MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "")
CTX = int(os.getenv("LLAMA_CTX", "2048"))

llm = None
if _HAS_LLAMA:
    if not MODEL_PATH:
        print("LLAMA_MODEL_PATH not set; set env var to path to .gguf")
    else:
        try:
            llm = Llama(model_path=MODEL_PATH, n_ctx=CTX)
            print("Loaded llama model:", MODEL_PATH)
        except Exception as e:
            print("Failed to load llama model:", e)
            llm = None

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 256

@app.post("/generate")
def generate(req: Req):
    if llm is None:
        raise HTTPException(status_code=500, detail="LLM not initialized or model failed to load")
    try:
        resp = llm(req.prompt, max_tokens=req.max_new_tokens)
        text = resp.get("choices", [{}])[0].get("text", "")
        return {"text": text}
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb.splitlines()[-20:]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)