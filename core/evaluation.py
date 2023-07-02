from human_eval.data import write_jsonl, read_problems
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
import itertools
import typing

BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], list[str]
]


def run_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    is_starcoder: bool = False,
):
    problems = read_problems()
    # problems = dict(itertools.islice(problems.items(), 20))
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    for task_id in problems:
        if is_starcoder:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task
        )

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)

    write_jsonl(out_path, samples)
