HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Federated ML Training Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .card { background: rgba(255,255,255,0.95); padding: 25px; margin: 15px 0; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }
        .status-badge { padding: 6px 16px; border-radius: 20px; color: white; font-size: 12px; font-weight: bold; display: inline-block; }
        .status-idle { background: linear-gradient(45deg, #6c757d, #5a6268); }
        .status-training { background: linear-gradient(45deg, #007bff, #0056b3); animation: pulse 2s infinite; }
        .status-completed { background: linear-gradient(45deg, #28a745, #1e7e34); }
        .status-failed { background: linear-gradient(45deg, #dc3545, #c82333); }
        .status-aggregating { background: linear-gradient(45deg, #ffc107, #e0a800); color: black; }
        .status-initializing { background: linear-gradient(45deg, #17a2b8, #138496); }
        
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        
        .btn { padding: 12px 24px; margin: 8px; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s ease; }
        .btn-primary { background: linear-gradient(45deg, #007bff, #0056b3); color: white; }
        .btn-danger { background: linear-gradient(45deg, #dc3545, #c82333); color: white; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        
        .worker-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        .worker-card { background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-left: 5px solid #007bff; border-radius: 10px; padding: 20px; }
        .progress-bar { width: 100%; height: 24px; background: #e9ecef; border-radius: 12px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(45deg, #007bff, #0056b3); transition: width 0.3s ease; }
        
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: linear-gradient(135deg, #17a2b8, #138496); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .metric-label { font-size: 0.9em; opacity: 0.9; }
        
        .connection-status { position: fixed; top: 20px; right: 20px; padding: 10px 15px; border-radius: 20px; color: white; font-weight: bold; }
        .connected { background: #28a745; }
        .disconnected { background: #dc3545; }
        
        .log-container { max-height: 300px; overflow-y: auto; background: #f8f9fa; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="connection-status" id="connection-status">Connecting...</div>
    
    <div class="container">
        <div class="header">
            <h1>🤖 Federated ML Training Dashboard</h1>
            <p>Distributed Training via Gossip Protocol</p>
        </div>
        
        <div class="card">
            <h2>🎮 Training Control</h2>
            <div id="training-status"></div>
            <div class="metrics" id="metrics"></div>
            <button class="btn btn-primary" id="start-btn" onclick="startTraining()">🚀 Start Training</button>
            <button class="btn btn-danger" id="stop-btn" onclick="stopTraining()">⏹️ Stop Training</button>
        </div>
        
        <div class="card">
            <h2>👥 Worker Nodes</h2>
            <div id="workers-container"></div>
        </div>
        
        <div class="card">
            <h2>📊 Training Progress</h2>
            <div id="progress-container"></div>
        </div>
        
        <div class="card">
            <h2>📝 System Logs</h2>
            <div class="log-container" id="log-container"></div>
        </div>
    </div>

    <script>
        let refreshInterval;
        let logMessages = [];
        const MAX_LOGS = 100;
        const TARGET_EPOCHS = 10; // Corresponds to master_server.py target_epochs

        function addLog(message, level = 'info') {
            const logContainer = document.getElementById('log-container');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.innerHTML = `<strong>[${timestamp}]</strong> ${message}`;
            logEntry.style.color = level === 'error' ? '#dc3545' : '#333';
            
            logMessages.push(logEntry.innerHTML);
            if (logMessages.length > MAX_LOGS) {
                logMessages.shift(); // Remove the oldest message
            }
            
            logContainer.innerHTML = logMessages.map(msg => `<div>${msg}</div>`).join('');
            logContainer.scrollTop = logContainer.scrollHeight;
        }

        function updateConnectionStatus(isConnected) {
            const statusEl = document.getElementById('connection-status');
            if (isConnected) {
                statusEl.textContent = 'Connected';
                statusEl.className = 'connection-status connected';
            } else {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'connection-status disconnected';
            }
        }

        function getStatusClass(status) {
            return `status-${(status || 'unknown').toLowerCase()}`;
        }
        
        function formatTimestamp(ts) {
            if (!ts) return 'N/A';
            const date = new Date(ts * 1000);
            return date.toLocaleString();
        }

        function updateStatusAndMetrics(data) {
            const statusEl = document.getElementById('training-status');
            statusEl.innerHTML = `Current Status: <span class="status-badge ${getStatusClass(data.training_status)}">${data.training_status.toUpperCase()}</span>`;
            
            const metricsEl = document.getElementById('metrics');
            metricsEl.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${data.current_round} / ${data.max_rounds}</div>
                    <div class="metric-label">Training Round</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${data.model_version}</div>
                    <div class="metric-label">Model Version</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${data.active_workers}</div>
                    <div class="metric-label">Active Workers</div>
                </div>
                 <div class="metric-card">
                    <div class="metric-value">${data.session_id ? data.session_id.substring(0, 8) + '...' : 'N/A'}</div>
                    <div class="metric-label">Session ID</div>
                </div>
            `;
            
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            const isTraining = ['TRAINING', 'AGGREGATING', 'INITIALIZING'].includes(data.training_status.toUpperCase());
            startBtn.disabled = isTraining;
            stopBtn.disabled = !isTraining;
        }
        
        function updateWorkers(workers) {
            const container = document.getElementById('workers-container');
            if (!workers || workers.length === 0) {
                container.innerHTML = '<p style="text-align: center;">No workers connected.</p>';
                return;
            }
            
            container.className = 'worker-grid';
            container.innerHTML = workers.map(worker => `
                <div class="worker-card" style="border-left-color: ${worker.status === 'failed' ? '#dc3545' : '#007bff'};">
                    <h4>Worker: ${worker.node_id}</h4>
                    <p><strong>Address:</strong> ${worker.host}:${worker.port}</p>
                    <p><strong>Status:</strong> <span class="status-badge ${getStatusClass(worker.status)}">${worker.status.toUpperCase()}</span></p>
                    <p><strong>Last Heartbeat:</strong> ${formatTimestamp(worker.last_heartbeat)}</p>
                    <p><strong>Current Epoch:</strong> ${worker.current_epoch}</p>
                    <p><strong>Loss:</strong> ${worker.loss.toFixed(4)}</p>
                </div>
            `).join('');
        }

        function updateProgress(workers, trainingStatus) {
            const container = document.getElementById('progress-container');
            const isTraining = ['TRAINING', 'AGGREGATING'].includes(trainingStatus.toUpperCase());
            const trainingWorkers = workers.filter(w => w.status !== 'failed' && isTraining);

            if (trainingWorkers.length === 0) {
                container.innerHTML = '<p style="text-align: center;">No workers are currently training.</p>';
                return;
            }
            
            container.innerHTML = trainingWorkers.map(worker => {
                const progress = worker.status === 'completed' ? 100 : (worker.current_epoch / TARGET_EPOCHS) * 100;
                return `
                    <div style="margin-bottom: 15px;">
                        <strong>${worker.node_id} - Epoch Progress</strong>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${progress.toFixed(2)}%;"></div>
                        </div>
                        <span>${worker.current_epoch} / ${TARGET_EPOCHS} Epochs</span>
                    </div>
                `
            }).join('');
        }

        async function updateDashboard() {
            try {
                const response = await fetch('/api/training_status');
                if (!response.ok) {
                    throw new Error(`HTTP Error: ${response.status}`);
                }
                const data = await response.json();
                
                updateConnectionStatus(true);
                updateStatusAndMetrics(data);
                updateWorkers(data.workers);
                updateProgress(data.workers, data.training_status);
                
            } catch (error) {
                updateConnectionStatus(false);
                console.error('Failed to update dashboard:', error);
            }
        }

        async function startTraining() {
            try {
                addLog('Sending request to start training...');
                const response = await fetch('/api/start_training', { method: 'POST' });
                const data = await response.json();
                
                if (response.ok) {
                    addLog(`Training started successfully! Session ID: ${data.session_id}`);
                } else {
                    throw new Error(data.detail || 'Unknown error');
                }
            } catch (error) {
                addLog(`Error starting training: ${error.message}`, 'error');
                console.error('Start training error:', error);
            }
        }
        
        async function stopTraining() {
            try {
                addLog('Sending request to stop training...');
                const response = await fetch('/api/stop_training', { method: 'POST' });
                const data = await response.json();

                if (response.ok) {
                    addLog('Training stopped successfully.');
                } else {
                    throw new Error(data.detail || 'Unknown error');
                }
            } catch (error) {
                addLog(`Error stopping training: ${error.message}`, 'error');
                console.error('Stop training error:', error);
            }
        }
        
        window.onload = function() {
            addLog("Dashboard loaded. Attempting to connect to master server.");
            updateDashboard(); // Initial call
            refreshInterval = setInterval(updateDashboard, 2000); // Poll every 2 seconds
        };
    </script>
</body>
</html>
"""
