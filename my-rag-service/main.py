import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from models import RunRequest, RunResponse, RunStatusResponse, RunResultResponse
from pipeline import app as graph_app

# Load environment variables
load_dotenv()

# In-memory storage for run status and results
# Structure: { run_id: { "status": "running"|"completed"|"failed", "result": ..., "error": ... } }
run_store: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 RAG Service Started")
    yield
    # Shutdown
    print("🛑 RAG Service Stopped")

app = FastAPI(
    title="Week 9 RAG Service",
    description="LangGraph Multi-Agent RAG Service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_graph(run_id: str, query: str, thread_id: str):
    """Background task to run the LangGraph pipeline."""
    try:
        run_store[run_id]["status"] = "running"
        
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {
            "messages": [HumanMessage(content=query)],
            "run_id": run_id, # Creating directory in pipeline
            "agent_results": {} # Initialize
        }
        
        # Invoke the graph
        output = await graph_app.ainvoke(inputs, config=config)
        
        # Extract final result (from writer mainly, or collection of results)
        final_doc = output.get("agent_results", {}).get("final_doc", "No final document produced.")
        
        run_store[run_id]["status"] = "completed"
        run_store[run_id]["result"] = {
            "final_doc": final_doc,
            "full_state": output.get("agent_results", {})
        }
        
    except Exception as e:
        print(f"❌ Error in run {run_id}: {e}")
        run_store[run_id]["status"] = "failed"
        run_store[run_id]["error"] = str(e)

@app.post("/api/v1/run", response_model=RunResponse)
async def submit_run(request: RunRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    run_store[run_id] = {"status": "pending"}
    
    background_tasks.add_task(process_graph, run_id, request.query, request.thread_id)
    
    return RunResponse(
        run_id=run_id,
        status="submitted",
        message="Request submitted successfully. Check status with /api/v1/status/{run_id}"
    )

@app.get("/api/v1/status/{run_id}", response_model=RunStatusResponse)
async def get_status(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run ID not found")
    
    state = run_store[run_id]
    return RunStatusResponse(
        run_id=run_id,
        status=state["status"],
        result=state.get("result"),
        logs=state.get("logs") # Placeholder if we implement log capture
    )

@app.get("/api/v1/result/{run_id}")
async def get_result(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run ID not found")
    
    state = run_store[run_id]
    if state["status"] != "completed":
         raise HTTPException(status_code=400, detail=f"Run is not completed. Current status: {state['status']}")
         
    return state["result"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
