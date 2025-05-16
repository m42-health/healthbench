import json
import multiprocessing as mp
from datasets import Dataset
from openai import OpenAI
import fire
import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm
import time

from prompts import SYSTEM_PROMPT


load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Assuming the model is hosted with OpenAI compatible api
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

MODEL_ID = ""


def is_server_running(port=8000):
    url = f"http://localhost:{port}/v1/models"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Server is running.")
            return True
    except requests.ConnectionError:
        pass
    return False


def generate_response(
    prompt: list,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.05,  # Low temperature for generation
    max_output_tokens: int = 4_000,
) -> list[dict]:
    global MODEL_ID
    model_name = MODEL_ID

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
    model_id: str = None,
):
    global MODEL_ID
    if model_id is not None:
        MODEL_ID = model_id  # In case a different model name is passed

    # Make sure local deployments dont contain gpt in their name
    if "gpt" in MODEL_ID:
        global client
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        # Wait for the server to be ready
        for _ in tqdm(
            range(300), desc="Waiting for the server to start!"
        ):  # Retry for up to ~300 seconds
            if is_server_running(port=8000):
                break
            time.sleep(5)
        else:
            print("Server did not start in time.")
            exit(1)

    INPUT_FILE_PATH = "data/benchmark/health_bench.jsonl"

    ds = load_dataset(input_filepath=INPUT_FILE_PATH)

    print(f"Generating reponses with {MODEL_ID}")
    ds = ds.map(lambda x: complete_turn(x), num_proc=mp.cpu_count() // 2)

    ds.to_json(os.path.join(output_dir, f"{MODEL_ID}.jsonl"), lines=True)


if __name__ == "__main__":
    fire.Fire(run)
