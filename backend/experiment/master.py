from fastapi import FastAPI, HTTPException
import uvicorn
from typing import List, Dict
import torch
import torch.nn as nn
from pydantic import BaseModel
import numpy as np
import asyncio
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
workers = {}
current_epoch = 0
model_state = None
is_training = False
MAX_EPOCHS = 10
epoch_updates = defaultdict(list)

class WorkerRegister(BaseModel):
    worker_id: str
    address: str
    port: int

class ModelUpdate(BaseModel):
    worker_id: str
    model_state: Dict
    epoch: int

def average_model_states(states_list):
    """Average multiple model states together."""
    averaged_state = {}
    for key in states_list[0].keys():
        # Convert lists back to tensors for averaging
        tensors = [torch.tensor(state[key]) for state in states_list]
        averaged_state[key] = torch.mean(torch.stack(tensors), dim=0).tolist()
    return averaged_state

async def check_epoch_completion():
    """Check if all workers have completed their current epoch and perform averaging."""
    global model_state, current_epoch, epoch_updates
    
    while is_training and current_epoch < MAX_EPOCHS:
        await asyncio.sleep(1)  # Check every second
        
        # Get all updates for current epoch
        current_updates = epoch_updates[current_epoch]
        
        # If all workers have reported for this epoch
        if len(current_updates) == len(workers):
            # Average the model states
            model_state = average_model_states(current_updates)
            
            # Clear updates for this epoch
            epoch_updates[current_epoch] = []
            
            # Increment epoch
            current_epoch += 1
            
            print(f"Completed epoch {current_epoch}/{MAX_EPOCHS}")
            
            # Notify all workers to start next epoch
            for worker_info in workers.values():
                try:
                    await notify_worker_next_epoch(worker_info["url"])
                except:
                    print(f"Failed to notify worker at {worker_info['url']}")
            
            if current_epoch >= MAX_EPOCHS:
                print("Training completed!")
                break

async def notify_worker_next_epoch(worker_url: str):
    """Notify a worker to start the next epoch."""
    import httpx
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"{worker_url}/start_epoch", json={
                "epoch": current_epoch,
                "model_state": model_state
            })
        except Exception as e:
            print(f"Error notifying worker {worker_url}: {e}")

@app.post("/register_worker")
async def register_worker(worker: WorkerRegister):
    worker_url = f"http://{worker.address}:{worker.port}"
    workers[worker.worker_id] = {
        "url": worker_url,
        "status": "idle"
    }
    return {"status": "registered", "total_workers": len(workers)}

@app.post("/start_training")
async def start_training():
    if len(workers) == 0:
        raise HTTPException(status_code=400, detail="No workers registered")
    
    asyncio.create_task(initialize_training())
    return {"status": "training_started", "workers": list(workers.keys())}

async def initialize_training():
    global is_training, current_epoch, model_state
    
    # Initialize model state if needed
    if model_state is None:
        # Create a simple model for demonstration
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        model_state = {k: v.tolist() for k, v in model.state_dict().items()}
    
    is_training = True
    current_epoch = 0
    
    # Start the epoch completion checker
    asyncio.create_task(check_epoch_completion())
    
    # Notify all workers to start training
    for worker_info in workers.values():
        try:
            await notify_worker_next_epoch(worker_info["url"])
        except:
            print(f"Failed to notify worker at {worker_info['url']}")

@app.post("/update_model")
async def update_model(update: ModelUpdate):
    global epoch_updates
    
    if not is_training:
        raise HTTPException(status_code=400, detail="Training not in progress")
    
    if update.worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not registered")
    
    # Store the update for the current epoch
    epoch_updates[update.epoch].append(update.model_state)
    
    return {
        "status": "update_accepted", 
        "current_epoch": current_epoch,
        "updates_received": len(epoch_updates[update.epoch])
    }


@app.get("/get_workers")
async def get_workers():
    return {"workers": list(workers.keys())}


@app.get("/get_model_state")
async def get_model_state():
    if model_state is None:
        raise HTTPException(status_code=404, detail="No model state available")
    return {
        "model_state": model_state, 
        "current_epoch": current_epoch,
        "is_training": is_training
    }

@app.get("/training_status")
async def get_training_status():
    return {
        "is_training": is_training,
        "current_epoch": current_epoch,
        "total_epochs": MAX_EPOCHS,
        "num_workers": len(workers),
        "updates_this_epoch": len(epoch_updates[current_epoch])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
