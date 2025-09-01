from fastapi import FastAPI, HTTPException
import uvicorn
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from typing import Dict
import numpy as np
from pydantic import BaseModel

class EpochData(BaseModel):
    epoch: int
    model_state: Dict

class Worker:
    def __init__(self, worker_id: str, master_url: str, host: str, port: int):
        self.app = FastAPI()
        self.worker_id = worker_id
        self.master_url = master_url
        self.host = host
        self.port = port
        self.model = None
        self.optimizer = None
        self.current_epoch = 0
        
        # Training data setup (simulated)
        self.setup_training_data()
        
        # Register routes
        self.setup_routes()
        
        # Register with master
        self.register_with_master()

    def setup_training_data(self):
        # Simulate different data for each worker
        # In real implementation, this would load actual data
        self.data_size = 1000
        self.batch_size = 32
        self.data = torch.randn(self.data_size, 10)  # Random features
        self.labels = torch.randint(0, 2, (self.data_size,))  # Random labels

    def setup_routes(self):
        @self.app.get("/status")
        async def get_status():
            return {
                "worker_id": self.worker_id,
                "status": "active",
                "current_epoch": self.current_epoch
            }

        @self.app.post("/start_epoch")
        async def start_epoch(data: EpochData):
            await self.train_epoch(data.epoch, data.model_state)
            return {"status": "epoch_completed", "epoch": data.epoch}

    def setup_model(self, state_dict):
        # Create model architecture
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Load state dict
        state_dict = {k: torch.tensor(v) for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        
        # Initialize optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    async def train_epoch(self, epoch: int, model_state: Dict):
        # Update model state
        if self.model is None:
            self.setup_model(model_state)
        else:
            state_dict = {k: torch.tensor(v) for k, v in model_state.items()}
            self.model.load_state_dict(state_dict)
        
        self.model.train()
        running_loss = 0.0
        
        # Train for one epoch
        for i in range(0, self.data_size, self.batch_size):
            batch_data = self.data[i:i+self.batch_size]
            batch_labels = self.labels[i:i+self.batch_size]
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        # Update master with new model state
        model_state = {k: v.tolist() for k, v in self.model.state_dict().items()}
        self.update_master(model_state, epoch)
        
        print(f"Worker {self.worker_id} completed epoch {epoch} with loss: {running_loss/len(self.data)}")
        self.current_epoch = epoch
    
    def register_with_master(self):
        response = requests.post(
            f"{self.master_url}/register_worker",
            json={
                "worker_id": self.worker_id,
                "address": self.host,
                "port": self.port
            }
        )
        if response.status_code != 200:
            raise Exception("Failed to register with master")
        print(f"Worker {self.worker_id} registered successfully")
    
    def update_master(self, model_state: Dict, epoch: int):
        response = requests.post(
            f"{self.master_url}/update_model",
            json={
                "worker_id": self.worker_id,
                "model_state": model_state,
                "epoch": epoch
            }
        )
        if response.status_code != 200:
            print(f"Failed to update master: {response.text}")
    
    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)

if __name__ == "__main__":
    import sys
    import random
    
    if len(sys.argv) != 2:
        print("Usage: python worker.py <worker_id>")
        sys.exit(1)
    
    worker_id = sys.argv[1]
    master_url = "http://localhost:8000"  # Adjust as needed
    host = "localhost"
    port = 8001 + random.randint(0, 999)  # Random port to allow multiple workers
    
    worker = Worker(worker_id, master_url, host, port)
    worker.run()
