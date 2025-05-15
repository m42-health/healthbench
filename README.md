# Healthbench - Standalone implementation

## Running server
```shell
# /home/praveen/miniconda3/envs/self-rewarding-llm/bin/python -m vllm.entrypoints.openai.api_server \
/home/praveen/miniconda3/envs/r1-training/bin/python -m vllm.entrypoints.openai.api_server \
        --model /models_llm/Qwen2.5-72B-Instruct \
        --port 8000 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --enable_prefix_caching \
        --served-model-name qwen2.5-72b

/home/praveen/miniconda3/envs/r1-training/bin/python -m vllm.entrypoints.openai.api_server \
        --model m42-health/Llama3-Med42-70B \
        --port 8000 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --enable_prefix_caching \
        --served-model-name med42-v2-70b

/home/praveen/miniconda3/envs/r1-training/bin/python -m vllm.entrypoints.openai.api_server \
        --model /models_llm/Llama-4-Maverick-17B-128E-Instruct \
        --port 8000 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --enable_prefix_caching \
        --max-model-len 16192 \
        --served-model-name llama4-maverick
```

## Generation

```shell
python generation.py --model_id="med42-v2-70b"
# python generation.py --model_id="llama4-maverick"
```

## Grading
```shell
python judge.py --input_data_path="data/generations/qwen2.5-72b.jsonl"
# python judge.py --input_data_path="data/generations/med42-v2-70b.jsonl"
# python judge.py --input_data_path="data/generations/llama4-maverick.jsonl"
```