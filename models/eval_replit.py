from human_eval.data import write_jsonl, read_problems
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import os
import torch
from tqdm import tqdm

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


def format_output(output: str):
    try:
        return output.replace("\t", "    ")
    except:
        return ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    prompt_input = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
{prompt}

### Response:"""

    input_batch = [prompt_input for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    output = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return [format_output(out) for out in output]


def run_eval(num_samples_per_task: int):
    problems = read_problems()

    tokenizer = AutoTokenizer.from_pretrained(
        "replit/replit-code-v1-3b",
        trust_remote_code=True,
        use_auth_token=TOKEN,
    )
    model = torch.compile(
        AutoModelForCausalLM.from_pretrained(
            "replit/replit-code-v1-3b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_auth_token=TOKEN,
            init_device="cuda",
        ).eval()
    )

    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)
    for task_id in problems:
        prompt = problems[task_id]["prompt"].replace("    ", "\t")
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

    write_jsonl("results/replit/eval.jsonl", samples)


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    os.makedirs("results/replit", exist_ok=True)

    run_eval(num_samples_per_task)
