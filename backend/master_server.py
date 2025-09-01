import asyncio
import json
import time
import uuid
import pickle
import base64
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

from utils import HTML_INTERFACE
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import websockets
from model import SimpleModel
from contextlib import asynccontextmanager

class MessageType(Enum):
    GOSSIP = "gossip"
    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    MODEL_UPDATE = "model_update"
    HEARTBEAT = "heartbeat"
    WORKER_JOIN = "worker_join"
    WORKER_LEAVE = "worker_leave"

class TrainingStatus(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing" 
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkerNode:
    node_id: str
    host: str
    port: int
    websocket_port: int
    status: str
    last_heartbeat: float
    current_epoch: int = 0
    loss: float = 0.0
    model_version: int = 0

@dataclass
class GossipMessage:
    message_id: str
    message_type: MessageType
    sender_id: str
    timestamp: float
    data: dict
    ttl: int = 3



class MasterServer:
    def __init__(self, port=8000):
        self.port = port
        self.workers: Dict[str, WorkerNode] = {}
        self.connected_websockets: Dict[str, WebSocket] = {}
        self.training_session_id: Optional[str] = None
        self.training_status = TrainingStatus.IDLE
        self.global_model = SimpleModel()
        self.model_version = 0
        self.target_epochs = 20
        self.current_round = 0
        self.max_rounds = 5
        self.seen_messages: Set[str] = set()
        
        self.message_queue = asyncio.Queue()
        
        self.app = FastAPI(title="Federated ML Master Server")
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.get("/")
        async def dashboard():
            return HTMLResponse(HTML_INTERFACE)
        
        @self.app.websocket("/ws/{node_id}")
        async def websocket_endpoint(websocket: WebSocket, node_id: str):
            await websocket.accept()
            self.connected_websockets[node_id] = websocket
            print(f"WebSocket connected: {node_id}")
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self.handle_websocket_message(node_id, message)
            except WebSocketDisconnect:
                print(f"WebSocket disconnected: {node_id}")
                if node_id in self.connected_websockets:
                    del self.connected_websockets[node_id]
                if node_id in self.workers:
                    self.workers[node_id].status = "failed"
        
        @self.app.get("/api/workers")
        async def get_workers():
            return [asdict(worker) for worker in self.workers.values()]
        
        @self.app.get("/api/training_status")
        async def get_training_status():
            return {
                'training_status': self.training_status.value,
                'session_id': self.training_session_id,
                'current_round': self.current_round,
                'max_rounds': self.max_rounds,
                'model_version': self.model_version,
                'active_workers': len([w for w in self.workers.values() if w.status != 'failed']),
                'workers': [asdict(w) for w in self.workers.values()]
            }
        
        @self.app.post("/api/start_training")
        async def start_training():
            if self.training_status != TrainingStatus.IDLE:
                raise HTTPException(status_code=400, detail="Training already in progress")
            
            active_workers = [w for w in self.workers.values() if w.status == 'idle']
            if not active_workers:
                raise HTTPException(status_code=400, detail="No active workers available")
            
            self.training_session_id = str(uuid.uuid4())
            self.training_status = TrainingStatus.INITIALIZING
            self.current_round = 0
            
            asyncio.create_task(self.run_federated_training())
            
            return {
                'status': 'training_started',
                'session_id': self.training_session_id,
                'workers_count': len(active_workers)
            }
        
        @self.app.post("/api/stop_training")
        async def stop_training():
            self.training_status = TrainingStatus.IDLE
            self.training_session_id = None
            
            stop_message = GossipMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.TRAINING_COMPLETE,
                sender_id="master",
                timestamp=time.time(),
                data={'action': 'stop', 'session_id': self.training_session_id}
            )
            await self.broadcast_gossip_message(stop_message)
            
            return {'status': 'training_stopped'}
    
    async def handle_websocket_message(self, node_id: str, message_data: dict):
        try:
            message_type = MessageType(message_data['type'])
            
            if message_type == MessageType.WORKER_JOIN:
                await self.handle_worker_join(node_id, message_data)
            elif message_type == MessageType.HEARTBEAT:
                await self.handle_heartbeat(node_id, message_data)
            elif message_type == MessageType.TRAINING_COMPLETE:
                await self.handle_training_complete(node_id, message_data)
            elif message_type == MessageType.GOSSIP:
                await self.handle_gossip_message(message_data)
                
        except Exception as e:
            print(f"Error handling message from {node_id}: {e}")
    
    async def handle_worker_join(self, node_id: str, data: dict):
        worker = WorkerNode(
            node_id=node_id,
            host=data['host'],
            port=data['port'],
            websocket_port=data['websocket_port'],
            status='idle',
            last_heartbeat=time.time()
        )
        self.workers[node_id] = worker
        print(f"Worker joined: {worker.host}:{worker.port}")
        
        gossip_msg = GossipMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.WORKER_JOIN,
            sender_id="master",
            timestamp=time.time(),
            data={'worker': asdict(worker)}
        )
        await self.broadcast_gossip_message(gossip_msg)
    
    async def handle_heartbeat(self, node_id: str, data: dict):
        if node_id in self.workers:
            worker = self.workers[node_id]
            worker.last_heartbeat = time.time()
            worker.status = data.get('status', worker.status)
            worker.current_epoch = data.get('current_epoch', worker.current_epoch)
            worker.loss = data.get('loss', worker.loss)
    
    async def handle_training_complete(self, node_id: str, data: dict):
        if node_id in self.workers:
            self.workers[node_id].status = 'completed'
            self.workers[node_id].model_version = data.get('model_version', 0)
            print(f"Worker {node_id} completed training")
    
    async def handle_gossip_message(self, message_data: dict):
        message = GossipMessage(**message_data)
        
        if message.message_id in self.seen_messages:
            return
        
        self.seen_messages.add(message.message_id)
        
        if message.message_type == MessageType.WORKER_LEAVE:
            worker_id = message.data.get('worker_id')
            if worker_id in self.workers:
                self.workers[worker_id].status = 'failed'
        
        if message.ttl > 0:
            message.ttl -= 1
            await self.broadcast_gossip_message(message, exclude_sender=message.sender_id)
    
    async def broadcast_gossip_message(self, message: GossipMessage, exclude_sender: str = None): # type: ignore
        message_dict = asdict(message)
        
        if 'message_type' in message_dict and isinstance(message_dict['message_type'], Enum):
            message_dict['message_type'] = message_dict['message_type'].value
        
        message_json = json.dumps(message_dict)
        
        for node_id, websocket in self.connected_websockets.items():
            if exclude_sender and node_id == exclude_sender:
                continue
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                print(f"Failed to send gossip to {node_id}: {e}")
    
    def serialize_model(self, model):
        buffer = pickle.dumps(model.state_dict())
        return base64.b64encode(buffer).decode('utf-8')
    
    def deserialize_model(self, model_data):
        buffer = base64.b64decode(model_data.encode('utf-8'))
        return pickle.loads(buffer)
    
    async def run_federated_training(self):
        try:
            active_workers = [w for w in self.workers.values() if w.status == 'idle']
            if not active_workers:
                self.training_status = TrainingStatus.IDLE
                return
            
            print(f"Starting federated training with {len(active_workers)} workers")
            
            for round_num in range(self.max_rounds):
                self.current_round = round_num + 1
                self.training_status = TrainingStatus.TRAINING
                
                print(f"\n--- Training Round {self.current_round}/{self.max_rounds} ---")
                
                self.model_version += 1
                training_message = GossipMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.TRAINING_START,
                    sender_id="master",
                    timestamp=time.time(),
                    data={
                        'session_id': self.training_session_id,
                        'model_data': self.serialize_model(self.global_model),
                        'model_version': self.model_version,
                        'epochs': self.target_epochs,
                        'round': self.current_round
                    }
                )
                await self.broadcast_gossip_message(training_message)
                
                await self.wait_for_training_completion()
                
                self.training_status = TrainingStatus.AGGREGATING
                await self.aggregate_models()
                
                print(f"Round {self.current_round} completed")
                await asyncio.sleep(2)
            
            self.training_status = TrainingStatus.COMPLETED
            print("Federated training completed!")
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.training_status = TrainingStatus.FAILED
    
    async def wait_for_training_completion(self):
        while True:
            active_workers = [w for w in self.workers.values() if w.status != 'failed']
            completed_workers = [w for w in active_workers if w.status == 'completed']
            
            if len(completed_workers) == len(active_workers) and len(active_workers) > 0:
                break
            
            await asyncio.sleep(2)
    
    async def aggregate_models(self):
        print("Aggregating models...")
        
        model_requests = GossipMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.MODEL_UPDATE,
            sender_id="master",
            timestamp=time.time(),
            data={'action': 'request_model', 'round': self.current_round}
        )
        await self.broadcast_gossip_message(model_requests)
        
        await asyncio.sleep(5)
        
        for worker in self.workers.values():
            if worker.status == 'completed':
                worker.status = 'idle'
                worker.current_epoch = 0
        
        print(f"Aggregated models for round {self.current_round}")
        
if __name__ == "__main__":
    server = MasterServer(port=8000)
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
