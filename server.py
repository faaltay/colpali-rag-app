from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.endpoints import pdf_ingest, query
from app.api.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- backend selection (ADDED) ---
STORAGE_MODE = os.getenv("STORAGE_MODE", "auto").lower()
LLM_MODE = os.getenv("LLM_MODE", "auto").lower()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

use_local_storage = False
if STORAGE_MODE == "local":
    use_local_storage = True
elif STORAGE_MODE == "auto" and not SUPABASE_KEY:
    use_local_storage = True

use_local_llm = False
if LLM_MODE == "local":
    use_local_llm = True
elif LLM_MODE == "auto" and not ANTHROPIC_API_KEY:
    use_local_llm = True

if use_local_storage:
    from storage.local_storage import upload_file, download_file, list_files, delete_file
    # You may need to adapt names where server used supabase client methods
else:
    # ...existing Supabase client init code...
    # keep existing supabase client import and setup
    pass

if use_local_llm:
    from llm.local_llm import LocalLLM
    llm = LocalLLM()
else:
    # ...existing Anthropic client init code...
    # ensure you have an object "llm" with method generate(prompt, **kwargs)
    pass

app.include_router(pdf_ingest.router)
app.include_router(query.router)
