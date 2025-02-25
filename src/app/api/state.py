from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from supabase.client import AsyncClient as SupabaseAsyncClient

from app.settings import Settings


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant.qdrant_url,
        api_key=settings.qdrant.qdrant_api_key,
    )


def create_supabase_client(settings: Settings) -> SupabaseAsyncClient:
    return SupabaseAsyncClient(
        supabase_key=settings.supabase.supabase_key,
        supabase_url=settings.supabase.supabase_url,
    )


def create_openai_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.openai.api_key, base_url=settings.openai.base_url
    )
