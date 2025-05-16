# Healthbench - Standalone implementation

## Running server
Any endpoint compatible with OpenAI's API can be used. In this example, we’re using the vLLM library to launch a server.

```shell
# This will lauch a server at default port 8000
python -m vllm.entrypoints.openai.api_server \
        --model m42-health/Llama3-Med42-70B \
        --port 8000 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --enable_prefix_caching \
        --served-model-name med42-v2-70b

```

## Generation
Use either the server or OpenAI models to generate responses for the HealthBench dataset. The generated responses will be saved in the `data/generations` directory.

```shell
python generation.py --model_id="med42-v2-70b"

# or
# python generation.py --model_id="gpt-4.1-mini"
```

## Grading
```shell
python judge.py --input_data_path="data/generations/qwen2.5-72b.jsonl"
# python judge.py --input_data_path="data/generations/med42-v2-70b.jsonl"
# python judge.py --input_data_path="data/generations/llama4-maverick.jsonl"
```