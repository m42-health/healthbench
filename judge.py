import re, os
import json
from openai import OpenAI
from datasets import Dataset
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import fire

from ast import literal_eval
from prompts import (
    GRADER_TEMPLATE,
    SYSTEM_PROMPT,
    HARD_PROMPT_IDS,
    CONSENSUS_PROMPT_IDS,
)

# Assuming the model is hosted with OpenAI compatible api
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

MODEL_ID = "llama3.1-70b-instruct"


def generate_response(
    prompt: list,
    system_prompt: str = SYSTEM_PROMPT,
    model_name: str = MODEL_ID,
    temperature: float = 0.1,  # Low temperature for judging
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


def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}


def backup_parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}. Backup parsing invoked!!")
        if (
            "true" in json_cleaned.lower()
        ):  # A hack to assign value to 'criteria_met' instead of a random choice
            print("Found criteria_met=true!")
            return {"criteria_met": True, "explanation": "None"}
        elif "false" in json_cleaned.lower():
            print("Found criteria_met=false!")
            return {"criteria_met": False, "explanation": "None"}
        else:
            return {"criteria_met": False, "explanation": "None"}


def calculate_score(grading_response_list: list[dict]) -> float | None:
    total_possible_points = sum(
        rubric_item["points"]
        for rubric_item in grading_response_list
        if rubric_item["points"] > 0
    )
    if total_possible_points == 0:
        # should not happen for overall score, but may happen for tags
        return None

    achieved_points = sum(
        rubric_item["points"]
        for rubric_item in grading_response_list
        if rubric_item["response_dict"]["criteria_met"]
    )
    overall_score = achieved_points / total_possible_points
    return overall_score


def grade_sample(example, max_retries=30):
    # if the input file has relevant keys, skipping grading
    if "score" in example.keys() and "metrics" in example.keys():
        return example

    convo_with_response = example["prompt"] + example["prompt_response"]

    def grade_rubric_item(rubric_item: str):
        convo_str = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in convo_with_response]
        )
        grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
            "<<rubric_item>>", str(rubric_item)
        )
        messages = [dict(content=grader_prompt, role="user")]

        counter = 0
        while counter < max_retries:
            grading_response = generate_response(messages)
            grading_response_dict = parse_json_to_dict(grading_response)
            if "criteria_met" in grading_response_dict:
                label = grading_response_dict["criteria_met"]
                if label is True or label is False:
                    break
            counter = counter + 1
            print("Grading failed due to bad JSON output, retrying...")

        if (counter > 0) and len(grading_response_dict) == 0:
            grading_response_dict = backup_parse_json_to_dict(grading_response)

        return grading_response_dict

    rubric_list = example["rubrics"]

    grading_response_list = [
        {
            "response_dict": grade_rubric_item(i["criterion"]),
            "points": i["points"],
            "tags": i["tags"],
        }
        for i in rubric_list
    ]

    # compute the overall score
    overall_score = calculate_score(grading_response_list)
    assert overall_score is not None
    metrics = {
        "overall_score": overall_score,
    }

    metrics = {
        "overall_score": overall_score,
    }

    # compute scores for example-level tags)
    example_tags = example["example_tags"]
    example_tag_scores = {tag: overall_score for tag in example_tags}
    assert len(example_tag_scores) == len(example_tags)  # No duplicates.
    metrics.update(example_tag_scores)

    rubric_tag_items_grades = defaultdict(list)
    for rubric_item in grading_response_list:
        grading_response = rubric_item["response_dict"]
        curr_item_tags = set()  # Ensure no duplicates in a rubric item.
        for tag in rubric_item["tags"]:
            rubric_tag_items_grades[tag].append(rubric_item)
            assert tag not in curr_item_tags
            curr_item_tags.add(tag)

    rubric_tag_scores = {}
    for tag, items_grades in rubric_tag_items_grades.items():
        # items, grades = zip(*items_grades)
        score = calculate_score(items_grades)
        if score is not None:  # implies at least one positive criterion
            rubric_tag_scores[tag] = score

    metrics.update(rubric_tag_scores)

    # construct the list of explanations and grades
    # Probably all this is not required!
    rubric_items_with_grades = []
    readable_explanation_list = []
    for rubric_item in grading_response_list:
        grading_response = rubric_item["response_dict"]
        explanation = grading_response.get("explanation", "No explanation provided")
        criteria_met = grading_response["criteria_met"]
        readable_explanation = (
            f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
        )
        readable_explanation_list.append(readable_explanation)
        rubric_items_with_grades.append(
            {
                **rubric_item,
            }
        )

    readable_explanation_list.sort(key=lambda x: x.startswith("[False]"), reverse=True)
    readable_explanation_str = "\n\n".join(readable_explanation_list)
    readable_explanation_str = f"\n\n{readable_explanation_str}"

    # breakpoint()

    return {
        "score": metrics["overall_score"],
        "metrics": str(metrics),
        "responses": str(grading_response_list),
        "prompt_id": example["prompt_id"],
    }


def load_dataset(input_filepath: str = None):
    with open(input_filepath) as f:
        data = [json.loads(line) for line in f]

    ds = Dataset.from_list(data)

    return ds


def _compute_clipped_stats(
    values: list,
    stat: str,
):
    """Computes the mean (clipped to [0, 1]), bootstrap std for that mean, and n_samples for final HealthBench scoring."""
    if stat == "mean":
        return np.clip(np.mean(values), 0, 1).item()
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        bootstrap_samples = [np.random.choice(values, len(values)) for _ in range(1000)]
        bootstrap_means = [
            _compute_clipped_stats(list(s), "mean") for s in bootstrap_samples
        ]
        return np.std(bootstrap_means).item()
    else:
        raise ValueError(f"Unknown {stat =}")


def _aggregate_get_clipped_mean(
    single_eval_results,
):
    """
    Aggregate multiple SingleEvalResults into a single EvalResult for HealthBench.
    For each metric, returns the stats in _compute_clipped_stats.
    """
    name2values = defaultdict(list)

    for single_eval_result in single_eval_results:
        metrics = literal_eval(single_eval_result["metrics"])
        for name, value in metrics.items():
            name2values[name].append(value)
        if single_eval_result["score"] is not None:
            name2values["score"].append(single_eval_result["score"])

    final_metrics = {}
    for name, values in name2values.items():
        for stat in ["mean", "n_samples", "bootstrap_std"]:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_clipped_stats(values, stat)

    return final_metrics


def process_and_save(subset_name: str, subset_ds: dict, output_dir: str):
    try:
        final_metrics = _aggregate_get_clipped_mean(subset_ds)
    except Exception as e:
        print(f"Error processing {subset_name}: {e}")
        breakpoint()

    print(f"{subset_name} final metrics:", final_metrics)

    with open(
        os.path.join(output_dir, f"final_metrics_{subset_name.lower()}.json"), "w"
    ) as f:
        json.dump(final_metrics, f, indent=2)


def run(
    input_data_path: str = None,
):

    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(input_data_path))[0]

    # Create target directory
    output_dir = os.path.join("data", base_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Files will be saved at {output_dir}")

    ds = load_dataset(input_filepath=input_data_path)

    metrics_ds = ds.map(lambda x: grade_sample(x), num_proc=mp.cpu_count())

    # ALL
    process_and_save("ALL", metrics_ds, output_dir=output_dir)

    # HARD
    hard_ds = metrics_ds.filter(
        lambda x: x["prompt_id"] in HARD_PROMPT_IDS,
        num_proc=mp.cpu_count() // 2,
        desc="Filtering HARD samples",
    )
    process_and_save("HARD", hard_ds, output_dir=output_dir)

    # CONSENSUS
    # TODO: Consensus filtering might not be as straight forward as HARD subset. It requires rubric filtering as well
    # consensus_ds = metrics_ds.filter(
    #     lambda x: x["prompt_id"] in CONSENSUS_PROMPT_IDS,
    #     num_proc=mp.cpu_count() // 2,
    #     desc="Filtering CONSENSUS samples",
    # )
    # process_and_save("CONSENSUS", consensus_ds, output_dir=output_dir)

    metrics_ds.to_json(os.path.join(output_dir, "judge_responses.jsonl"), lines=True)


if __name__ == "__main__":

    fire.Fire(run)
