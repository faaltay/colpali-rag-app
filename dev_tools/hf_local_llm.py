from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import traceback

app = FastAPI()

# small CPU model for local testing
GENERATOR = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2", device=-1)

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.post("/generate")
def generate(req: Req):
    try:
        out = GENERATOR(req.prompt, max_new_tokens=req.max_new_tokens, do_sample=False)
        text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
        return {"text": text}
    except Exception as e:
        tb = traceback.format_exc()
        # print to server log and return controlled HTTPException JSON
        print(tb)
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb.splitlines()[-20:]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)