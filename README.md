# HealthBench - Standalone Implementation

This repository provides a standalone implementation for evaluating large language models (LLMs) using OpenAI's HealthBench dataset. HealthBench is a benchmark designed to assess the performance and safety of LLMs in realistic healthcare scenarios.

This repository is adapted from [original implementation](https://github.com/openai/simple-evals/) by OpenAI.

## Overview

HealthBench consists of 5,000 multi-turn conversations between users (patients or clinicians) and AI models, covering a wide range of medical topics and scenarios. Each conversation is accompanied by a set of physician-created rubric criteria, totaling over 48,562 unique items, to grade model responses based on accuracy, relevance, and safety.


## Running the Server

You can launch a server compatible with OpenAI's API using the vLLM library. This allows for local inference with your chosen model.

```bash
# Launches a server at default port 8000
python -m vllm.entrypoints.openai.api_server \
        --model m42-health/Llama3-Med42-70B \
        --port 8000 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --enable_prefix_caching \
        --served-model-name med42-v2-70b
```

## Generating Responses

Use the provided `generation.py` script to generate model responses for the HealthBench dataset. Responses will be saved in the `data/generations` directory.

```bash
python generation.py --model_id="med42-v2-70b"

# Alternatively, to use an OpenAI model:
# python generation.py --model_id="gpt-4.1-mini"
```

## Grading Responses

Evaluate the generated responses using the `judge.py` script. This script compares model outputs against the physician-created rubrics and computes detailed evaluation metrics.
Grading responses requires a model to be available locally or hosted remotely for evaluating against the rubrics. You may need to modify `judge.py` to match your model's API or configuration.
```bash
python judge.py --input_data_path="data/generations/med42-v2-70b.jsonl"
```

This will generate two output files in the working directory:

- `full_metrics_all.json`: Contains overall scores and detailed results across all themes and rubric axes in the complete HealthBench dataset.
- `full_metrics_hard.json`: Contains results focused on the more challenging subset of examples (HealthBench Hard).

These files provide granular insights into how the model performs in both general and difficult medical scenarios.


## Dataset Details

- **Conversations**: 5,000 realistic health dialogues, both synthetic and adversarially generated.
- **Rubric Criteria**: Over 48,562 unique items created by 262 physicians from 60 countries.
- **Evaluation Dimensions**: Accuracy, instruction following, communication quality, and more.
- **Special Sets**:
  - *HealthBench Consensus*: Focuses on 34 critical dimensions validated by physician consensus.
  - *HealthBench Hard*: Contains challenging examples where current top models score around 32%.

For more information, refer to the [HealthBench paper](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf) and the [OpenAI blog post](https://openai.com/index/healthbench/).

## License

This project is licensed under CC BY-NC-4.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Citation
This implementation is part of MEDIC evaluation framework. If you find this repository useful, please consider giving a star and citation:
```
@article{kanithi2024medic,
  title={Medic: Towards a comprehensive framework for evaluating llms in clinical applications},
  author={Kanithi, Praveen K and Christophe, Cl{\'e}ment and Pimentel, Marco AF and Raha, Tathagata and Saadi, Nada and Javed, Hamza and Maslenkova, Svetlana and Hayat, Nasir and Rajan, Ronnie and Khan, Shadab},
  journal={arXiv preprint arXiv:2409.07314},
  year={2024}
}
```
