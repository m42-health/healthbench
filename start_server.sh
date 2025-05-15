
# /home/praveen/miniconda3/envs/self-rewarding-llm/bin/python -m vllm.entrypoints.openai.api_server \
/home/praveen/miniconda3/envs/r1-training/bin/python -m vllm.entrypoints.openai.api_server \
        --model /models_llm/Qwen2.5-72B-Instruct \
        --port 8000 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --enable_prefix_caching \
        --served-model-name qwen2.5-72b