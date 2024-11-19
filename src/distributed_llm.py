import torch.distributed as dist
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import socket
import json
import psutil
import GPUtil
from flask import Flask, request, jsonify
import requests
import os
import logging
from datetime import datetime
import time
from typing import Dict, List, Optional
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distributed_llm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    @staticmethod
    def get_available_memory() -> Dict[str, float]:
        """Get available CPU and GPU memory in GB"""
        # CPU Memory
        cpu_memory = psutil.virtual_memory().available / (1024**3)  # Convert to GB
        
        # GPU Memory (if available)
        try:
            gpus = GPUtil.getGPUs()
            gpu_memory = gpus[0].memoryFree / 1024 if gpus else 0  # Convert to GB
        except Exception as e:
            logger.warning(f"Unable to get GPU memory: {e}")
            gpu_memory = 0
            
        return {
            "cpu": cpu_memory,
            "gpu": gpu_memory
        }

    @staticmethod
    def estimate_layer_memory(model_name: str) -> float:
        """Estimate memory required per transformer layer in GB"""
        # These are approximate values - adjust based on actual model
        layer_sizes = {
            "llama2-7b": 0.6,
            "llama2-13b": 1.1,
            "llama2-50b": 4.2
        }
        base_model = model_name.split('/')[-1].lower()
        return layer_sizes.get(base_model, 2.0)  # Default to 2GB if unknown

class HealthCheck:
    def __init__(self, check_interval: int = 30):
        self.last_heartbeat = datetime.now()
        self.is_healthy = True
        self.check_interval = check_interval
        self.lock = threading.Lock()
        
    def update_heartbeat(self):
        with self.lock:
            self.last_heartbeat = datetime.now()
            
    def start_monitoring(self):
        def monitor():
            while True:
                with self.lock:
                    time_since_heartbeat = (datetime.now() - self.last_heartbeat).seconds
                    self.is_healthy = time_since_heartbeat < self.check_interval
                time.sleep(self.check_interval / 2)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

class DistributedLLMNode:
    def __init__(
        self,
        model_name: str,
        node_id: int,
        total_nodes: int,
        master_ip: str = "localhost",
        master_port: int = 29500,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        self.model_name = model_name
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.master_ip = master_ip
        self.master_port = master_port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.health_check = HealthCheck()
        
        self.initialize_node()
        self.health_check.start_monitoring()
        
    def initialize_node(self):
        """Initialize the node with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                self._setup_distributed()
                self._load_model()
                self._partition_model()
                logger.info(f"Node {self.node_id} initialized successfully")
                return
            except Exception as e:
                logger.error(f"Initialization attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Failed to initialize node after {self.max_retries} attempts")

    def _setup_distributed(self):
        """Set up distributed environment with error handling"""
        os.environ['MASTER_ADDR'] = self.master_ip
        os.environ['MASTER_PORT'] = str(self.master_port)
        try:
            dist.init_process_group("gloo", rank=self.node_id, world_size=self.total_nodes)
        except Exception as e:
            logger.error(f"Failed to initialize distributed process group: {e}")
            raise

    def _load_model(self):
        """Load model with memory optimization"""
        memory_monitor = MemoryMonitor()
        available_memory = memory_monitor.get_available_memory()
        
        # Configure model loading based on available memory
        if available_memory['gpu'] > 2.0:  # Arbitrary threshold
            device_map = "auto"
            dtype = torch.float16
        else:
            device_map = "cpu"
            dtype = torch.float32
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _partition_model(self):
        """Partition model based on available memory across nodes"""
        memory_monitor = MemoryMonitor()
        layer_memory = memory_monitor.estimate_layer_memory(self.model_name)
        available_memory = memory_monitor.get_available_memory()
        
        # Calculate optimal layer distribution
        total_layers = len(self.model.layers)
        node_memories = self._gather_node_memories()
        
        # Distribute layers proportionally to available memory
        layer_assignments = self._calculate_layer_distribution(
            total_layers, node_memories, layer_memory
        )
        
        # Keep only assigned layers
        start_idx = sum(layer_assignments[:self.node_id])
        end_idx = start_idx + layer_assignments[self.node_id]
        self.model.layers = self.model.layers[start_idx:end_idx]
        
        logger.info(f"Node {self.node_id} assigned layers {start_idx} to {end_idx}")

    def _gather_node_memories(self) -> List[float]:
        """Gather available memory information from all nodes"""
        local_memory = MemoryMonitor.get_available_memory()['cpu']
        memory_tensor = torch.tensor([local_memory], dtype=torch.float32)
        
        gathered_memories = [torch.zeros_like(memory_tensor) for _ in range(self.total_nodes)]
        dist.all_gather(gathered_memories, memory_tensor)
        
        return [tensor.item() for tensor in gathered_memories]

    def _calculate_layer_distribution(
        self,
        total_layers: int,
        node_memories: List[float],
        layer_memory: float
    ) -> List[int]:
        """Calculate optimal layer distribution based on available memory"""
        total_memory = sum(node_memories)
        base_distribution = [
            int(memory / total_memory * total_layers)
            for memory in node_memories
        ]
        
        # Adjust for rounding errors
        remaining_layers = total_layers - sum(base_distribution)
        for i in range(remaining_layers):
            base_distribution[i] += 1
            
        return base_distribution

    def process_input(self, input_text: str) -> Dict:
        """Process input through this node's portion of the model with error handling"""
        try:
            self.health_check.update_heartbeat()
            
            inputs = self.tokenizer(input_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Synchronize with other nodes
            dist.all_reduce(outputs.logits)
            
            return {
                "status": "success",
                "outputs": outputs,
                "node_id": self.node_id
            }
        except Exception as e:
            logger.error(f"Error processing input on node {self.node_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "node_id": self.node_id
            }

    def cleanup(self):
        """Cleanup resources"""
        try:
            dist.destroy_process_group()
            if hasattr(self, 'model'):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

class DistributedLLMAPI:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.app = Flask(__name__)
        self.node: Optional[DistributedLLMNode] = None
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            if not self.node or not self.node.health_check.is_healthy:
                return jsonify({"status": "unhealthy"}), 503
            return jsonify({"status": "healthy"})
        
        @self.app.route('/initialize', methods=['POST'])
        def initialize():
            try:
                data = request.json
                self.node = DistributedLLMNode(
                    model_name=data['model_name'],
                    node_id=data['node_id'],
                    total_nodes=data['total_nodes'],
                    master_ip=data.get('master_ip', 'localhost'),
                    master_port=data.get('master_port', 29500)
                )
                return jsonify({"status": "initialized"})
            except Exception as e:
                logger.error(f"Initialization error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/generate', methods=['POST'])
        def generate():
            if not self.node:
                return jsonify({"error": "Node not initialized"}), 400
                
            try:
                data = request.json
                result = self.node.process_input(data['input_text'])
                
                if result['status'] == 'error':
                    return jsonify(result), 500
                    
                return jsonify({
                    "output": self.node.tokenizer.decode(
                        result['outputs'].logits.argmax(dim=-1).squeeze()
                    ),
                    "node_id": result['node_id']
                })
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/shutdown', methods=['POST'])
        def shutdown():
            if self.node:
                self.node.cleanup()
            return jsonify({"status": "shutdown complete"})
    
    def run(self, host: str = "0.0.0.0", port: int = 5000):
        self.app.run(host=host, port=port)

def setup_cluster(
    model_name: str,
    raspberry_pi_ips: List[str],
    master_ip: str,
    base_port: int = 5000
) -> bool:
    """Setup the distributed LLM cluster with health checking"""
    try:
        # Initialize each node
        for i, pi_ip in enumerate(raspberry_pi_ips):
            response = requests.post(
                f"http://{pi_ip}:{base_port}/initialize",
                json={
                    "model_name": model_name,
                    "node_id": i,
                    "total_nodes": len(raspberry_pi_ips),
                    "master_ip": master_ip,
                    "master_port": 29500
                },
                timeout=300  # 5-minute timeout for initialization
            )
            response.raise_for_status()
            
        # Verify health of all nodes
        for pi_ip in raspberry_pi_ips:
            response = requests.get(f"http://{pi_ip}:{base_port}/health")
            if response.status_code != 200:
                raise RuntimeError(f"Node {pi_ip} is unhealthy")
                
        logger.info("Cluster setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Cluster setup failed: {e}")
        # Attempt to shutdown any initialized nodes
        for pi_ip in raspberry_pi_ips:
            try:
                requests.post(f"http://{pi_ip}:{base_port}/shutdown")
            except:
                pass
        return False

# Example usage
if __name__ == "__main__":
    # If this is a worker node
    api = DistributedLLMAPI()
    api.run()
