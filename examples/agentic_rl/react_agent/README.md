# Training ReAct Agent using RL with Trinity-RFT

> [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) is a flexible, general-purpose framework for reinforcement fine-tuning (RFT) of large language models (LLMs). It provides high compatibility with agent frameworks and supports training agents without code modification.

In this example, we demonstrate how to train an agent built by AgentScope in Trinity-RFT.

## Preparation

1. Make sure you have at least 2 NVIDIA GPU with CUDA 12.4 or above installed. Here we use 8 H20 GPUs for training, you can adjust the configuration based on your hardware.

2. Install Trinity-RFT through its [Installation Guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html)


3. Download the GSM8K dataset and Qwen/Qwen3-8B model checkpoints.

```bash
huggingface-cli download openai/gsm8k --repo-type dataset
huggingface-cli download Qwen/Qwen3-8B
```

## Run the Example

You can run the example with the following command:

```bash
# setup ray cluster
ray start --head

# run the training script
python train_react.py
```
