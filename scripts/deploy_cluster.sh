#!/bin/bash
# Configuration
NODES=(
    "pi@192.168.1.101"
    "pi@192.168.1.102"
    "pi@192.168.1.103"
)

# Deploy to each node
deploy_node() {
    local node=$1
    echo "Deploying to $node..."
    
    # Copy installation files
    scp -r src scripts config "$node:~/llm_cluster/"
    
    # Run installation
    ssh "$node" "cd ~/llm_cluster && \
                 chmod +x scripts/*.sh && \
                 ./scripts/install_requirements.sh && \
                 ./scripts/setup_node.sh && \
                 sudo cp config/llm_node.service /etc/systemd/system/ && \
                 sudo systemctl daemon-reload && \
                 sudo systemctl enable llm_node && \
                 sudo systemctl start llm_node"
}

# Main deployment
main() {
    echo "Starting cluster deployment..."
    
    # Deploy to each node in parallel
    for node in "${NODES[@]}"; do
        deploy_node "$node" &
    done
    
    # Wait for all deployments to complete
    wait
    
    echo "Deployment completed. Checking node status..."
    
    # Check status of each node
    for node in "${NODES[@]}"; do
        echo "Status for $node:"
        ssh "$node" "sudo systemctl status llm_node --no-pager"
    done
}

main
