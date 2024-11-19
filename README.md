# Distributed LLM Cluster

This is experimental code. The solution will look at each nodes resources(GPU/CPU) and decide on how much of the LLM each node will look after. 

The second phase will be to present the clustered LLM as a single interface to be used within ollama.



This package contains scripts and configuration files for setting up a distributed LLM system across multiple Raspberry Pi nodes.

## Directory Structure

```
distributed-llm-cluster/
├── src/
│   └── distributed_llm.py
├── scripts/
│   ├── install_requirements.sh
│   ├── setup_node.sh
│   └── deploy_cluster.sh
├── config/
│   ├── config.json
│   └── llm_node.service
├── examples/
│   └── example_usage.py
└── README.md
```

## Installation

1. Update the node IPs in `config/config.json` and `scripts/deploy_cluster.sh`
2. Run the deployment script:
   ```bash
   chmod +x scripts/deploy_cluster.sh
   ./scripts/deploy_cluster.sh
   ```

## Usage

See `examples/example_usage.py` for example usage of the distributed LLM cluster.

## Monitoring

Check service status:
```bash
sudo systemctl status llm_node
```

View logs:
```bash
sudo journalctl -u llm_node -f
```
