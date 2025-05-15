import json
import multiprocessing as mp
from datasets import Dataset
from openai import OpenAI
import fire
import os

from prompts import SYSTEM_PROMPT

# Assuming the model is hosted with OpenAI compatible api
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

MODEL_ID = "qwen2.5-72b"


def generate_response(
    prompt: list,
    system_prompt: str = SYSTEM_PROMPT,
    model_name: str = MODEL_ID,
    temperature: float = 0.05,  # Low temperature for generation
    max_output_tokens: int = 4_000,
) -> list[dict]:
    # generate response
    message_list = [{"role": "system", "content": system_prompt}] + prompt
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
        messages=message_list,
    )

    return response.choices[0].message.content


def complete_turn(example: dict) -> dict:
    example["prompt_response"] = [
        {"role": "assistant", "content": generate_response(example["prompt"])}
    ]
    # print(example["prompt_response"][0]["content"])
    # breakpoint()
    return example


def load_dataset(input_filepath="data/health_bench.jsonl"):
    with open(input_filepath) as f:
        data = [json.loads(line) for line in f]

    ds = Dataset.from_list(data)

    return ds


def run(
    output_dir: str = "data/generations",
    model_id: str = MODEL_ID,
):
    global MODEL_ID
    MODEL_ID = model_id  # In case a different model name is passed

    INPUT_FILE_PATH = "data/benchmark/health_bench.jsonl"

    ds = load_dataset(input_filepath=INPUT_FILE_PATH)

    ds = ds.map(lambda x: complete_turn(x), num_proc=mp.cpu_count() // 2)

    ds.to_json(os.path.join(output_dir, f"{MODEL_ID}.jsonl"), lines=True)


if __name__ == "__main__":
    fire.Fire(run)
