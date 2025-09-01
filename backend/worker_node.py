import asyncio
import json
import time
import uuid
import pickle
import base64
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import uvicorn
from fastapi import FastAPI, WebSocket
import websockets
import numpy as np
import threading
from typing import Dict, Set
from master_server import MessageType

from model import SimpleModel

class WorkerNode:
    def __init__(self, node_id: str, master_host='localhost', master_port=8000, 
                 worker_port=8002, websocket_port=8003):
        self.node_id = node_id
        self.master_host = master_host
        self.master_port = master_port
        self.worker_port = worker_port
        self.websocket_port = websocket_port
        
        self.model = SimpleModel()
        self.status = 'idle'
        self.current_epoch = 0
        self.current_loss = 0.0
        self.model_version = 0
        self.training_session_id = None
        
        self.peer_workers: Dict[str, dict] = {}
        self.seen_messages: Set[str] = set()
        
        self.train_data = self.generate_dummy_data()
        
        self.master_websocket = None
        
        self.app = FastAPI(title=f"Worker Node {self.node_id}")
        self.setup_routes()
    
    def generate_dummy_data(self, num_samples=1000):
        X = torch.randn(num_samples, 784)
        y = torch.randint(0, 10, (num_samples,))
        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    
    def setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self.handle_gossip_message(message)
            except Exception as e:
                print(f"WebSocket error: {e}")
            
        @self.app.get("/status")
        async def get_status():
            return {
                'node_id': self.node_id,
                'status': self.status,
                'current_epoch': self.current_epoch,
                'loss': self.current_loss,
                'model_version': self.model_version,
                'peer_count': len(self.peer_workers)
            }
        
        @self.app.get("/peers")
        async def get_peers():
            return list(self.peer_workers.values())
    
    async def connect_to_master(self):
        while True:
            try:
                uri = f"ws://{self.master_host}:{self.master_port}/ws/{self.node_id}"
                self.master_websocket = await websockets.connect(uri)
                print(f"Connected to master at {uri}")
                
                join_message = {
                    'type': MessageType.WORKER_JOIN.value,
                    'node_id': self.node_id,
                    'host': 'localhost',
                    'port': self.worker_port,
                    'websocket_port': self.websocket_port,
                    'timestamp': time.time()
                }
                await self.master_websocket.send(json.dumps(join_message))
                
                listen_task = asyncio.create_task(self.listen_to_master())
                heartbeat_task = asyncio.create_task(self.send_heartbeat())
                await asyncio.gather(listen_task, heartbeat_task)

            except Exception as e:
                print(f"Failed to connect to master: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
    
    async def listen_to_master(self):
        try:
            async for message in self.master_websocket: # type: ignore
                data = json.loads(message)
                await self.handle_gossip_message(data)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection to master lost: {e}")
        except Exception as e:
            print(f"Error listening to master: {e}")
    
    async def send_heartbeat(self):
        while True:
            if not self.master_websocket:
                print("Master websocket not available. Heartbeat paused.")
                break

            try:
                heartbeat = {
                    'type': MessageType.HEARTBEAT.value,
                    'node_id': self.node_id,
                    'status': self.status,
                    'current_epoch': self.current_epoch,
                    'loss': self.current_loss,
                    'model_version': self.model_version,
                    'timestamp': time.time()
                }
                await self.master_websocket.send(json.dumps(heartbeat))

            except websockets.exceptions.ConnectionClosed:
                print("Heartbeat failed: Connection is closed.")
                break
            except Exception as e:
                print(f"Heartbeat failed with an unexpected error: {e}")
                break

            await asyncio.sleep(10)
    
    async def handle_gossip_message(self, message_data: dict):
        try:
            if 'message_id' in message_data and message_data['message_id'] in self.seen_messages:
                return
            
            if 'message_id' in message_data:
                self.seen_messages.add(message_data['message_id'])
            
            message_type_str = message_data.get('type') or message_data.get('message_type')
            if not message_type_str:
                return 
            message_type = MessageType(message_type_str)
            
            if message_type == MessageType.TRAINING_START:
                await self.handle_training_start(message_data)
            elif message_type == MessageType.WORKER_JOIN:
                await self.handle_peer_join(message_data)
            elif message_type == MessageType.MODEL_UPDATE:
                await self.handle_model_request(message_data)
            
            if message_data.get('ttl', 0) > 0:
                message_data['ttl'] -= 1
                await self.propagate_gossip(message_data)
                
        except Exception as e:
            print(f"Error handling gossip message: {e}")
    
    async def handle_training_start(self, data: dict):
        if self.status == 'idle' and data['data']['session_id']:
            print(f"Received training start command for session {data['data']['session_id']}")
            self.training_session_id = data['data']['session_id']
            self.model_version = data['data']['model_version']
            
            model_data = data['data']['model_data']
            buffer = base64.b64decode(model_data.encode('utf-8'))
            state_dict = pickle.loads(buffer)
            self.model.load_state_dict(state_dict)
            
            epochs = data['data']['epochs']
            asyncio.create_task(self.train_model(epochs))
    
    async def handle_peer_join(self, data: dict):
        worker_info = data['data']['worker']
        peer_id = worker_info['node_id']
        if peer_id != self.node_id:
            self.peer_workers[peer_id] = worker_info
            print(f"Peer worker joined: {peer_id}")
    
    async def handle_model_request(self, data: dict):
        if data['data']['action'] == 'request_model' and self.status == 'completed':
            print(f"Received model request for round {data['data']['round']}. Sending model to master.")
            model_data = self.serialize_model(self.model)
            response_message = {
                'type': MessageType.MODEL_UPDATE.value,
                'node_id': self.node_id,
                'data': {
                    'action': 'model_response',
                    'model_data': model_data,
                    'model_version': self.model_version,
                    'round': data['data']['round']
                },
                'timestamp': time.time()
            }
            if self.master_websocket:
                await self.master_websocket.send(json.dumps(response_message))
    
    async def propagate_gossip(self, message_data: dict):
        for peer_id, peer_info in self.peer_workers.items():
            try:
                pass
            except Exception as e:
                print(f"Failed to propagate gossip to {peer_id}: {e}")
    
    def serialize_model(self, model):
        buffer = pickle.dumps(model.state_dict())
        return base64.b64encode(buffer).decode('utf-8')
    
    async def train_model(self, epochs):
        print(f"Starting local training for {epochs} epochs")
        self.status = 'training'
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in self.train_data:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            self.current_loss = epoch_loss / num_batches
            print(f"Worker {self.node_id} - Epoch {self.current_epoch}, Loss: {self.current_loss:.4f}")
            
            await asyncio.sleep(0.5)
        
        self.status = 'completed'
        
        completion_message = {
            'type': MessageType.TRAINING_COMPLETE.value,
            'node_id': self.node_id,
            'data': {
                'session_id': self.training_session_id,
                'model_version': self.model_version,
                'final_loss': self.current_loss
            },
            'timestamp': time.time()
        }
        if self.master_websocket:
            await self.master_websocket.send(json.dumps(completion_message))
        
        print(f"Worker {self.node_id} completed training")
        
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a federated learning worker node.")
    parser.add_argument("--node-id", required=True, help="Unique ID for this worker node.")
    parser.add_argument("--port", type=int, required=True, help="HTTP port for the worker's API.")
    parser.add_argument("--ws-port", type=int, required=True, help="WebSocket port for the worker's gossip protocol.")
    parser.add_argument("--master-host", default="localhost", help="Host of the master server.")
    parser.add_argument("--master-port", type=int, default=8001, help="WebSocket port of the master server.")
    args = parser.parse_args()

    worker = WorkerNode(
        node_id=args.node_id,
        master_host=args.master_host,
        master_port=args.master_port,
        worker_port=args.port,
        websocket_port=args.ws_port
    )

    def run_api():
        uvicorn.run(worker.app, host="0.0.0.0", port=args.port)

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    try:
        asyncio.run(worker.connect_to_master())
    except KeyboardInterrupt:
        print(f"Worker {args.node_id} shutting down.")
