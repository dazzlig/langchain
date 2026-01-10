from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class RunRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default_thread"

class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str

class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    logs: Optional[List[str]] = None

class RunResultResponse(BaseModel):
    result: Dict[str, Any]
