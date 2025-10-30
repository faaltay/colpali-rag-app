# Minimal local storage backend replacing supabase storage calls used by the app.
from pathlib import Path
import shutil
from typing import List

ROOT = Path(__file__).parent.parent / "local_storage_data"
ROOT.mkdir(parents=True, exist_ok=True)

def upload_file(src_path: str, dest_name: str) -> str:
    """
    Save src_path into local storage and return a file:// URL or relative path.
    """
    dst = ROOT / dest_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dst)
    return f"file://{dst.resolve()}"

def download_file(dest_name: str, out_path: str) -> str:
    src = ROOT / dest_name
    if not src.exists():
        raise FileNotFoundError(dest_name)
    shutil.copy(src, out_path)
    return out_path

def list_files(prefix: str = "") -> List[str]:
    files = []
    for p in ROOT.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(ROOT).as_posix())
            if rel.startswith(prefix):
                files.append(rel)
    return files

def delete_file(name: str) -> bool:
    p = ROOT / name
    if p.exists():
        p.unlink()
        return True
    return False