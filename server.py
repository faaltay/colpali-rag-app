from fastapi import FastAPI

from app.api.endpoints import pdf_ingest, query
from app.api.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(pdf_ingest.router)
app.include_router(query.router)
