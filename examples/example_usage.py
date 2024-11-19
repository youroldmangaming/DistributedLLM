import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    with open('../config/config.json', 'r') as f:
        return json.load(f)

def initialize_cluster(config):
    """Initialize all nodes in the cluster"""
    for node in config['nodes']:
        try:
            response = requests.post(
                f"http://{node['ip']}:{node['port']}/initialize",
                json={
                    "model_name": config['model_name'],
                    "node_id": node['id'],
                    "total_nodes": len(config['nodes']),
                    "master_ip": config['master_ip'],
                    "master_port": config['master_port']
                },
                timeout=300
            )
            response.raise_for_status()
            logger.info(f"Node {node['id']} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize node {node['id']}: {e}")
            return False
    return True

def generate_text(prompt, config):
    """Generate text using the distributed cluster"""
    try:
        response = requests.post(
            f"http://{config['master_ip']}:{config['base_port']}/generate",
            json={"input_text": prompt},
            timeout=60
        )
        response.raise_for_status()
        return response.json()['output']
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return None

def main():
    # Load configuration
    config = load_config()
    
    # Initialize cluster
    if not initialize_cluster(config):
        logger.error("Cluster initialization failed")
        return
    
    # Wait for all nodes to be ready
    time.sleep(10)
    
    # Test generation
    prompt = "Here is a story about a creative computer:"
    result = generate_text(prompt, config)
    
    if result:
        logger.info(f"Generated text: {result}")
    else:
        logger.error("Text generation failed")

if __name__ == "__main__":
    main()
