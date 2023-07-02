from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, GPTBigCodeForCausalLM, PreTrainedTokenizer
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
    model: GPTBigCodeForCausalLM, tokenizer, prompt, batch_size
) -> list[str]:
    batch_input = [tokenize_opencode(tokenizer, prompt) for _ in range(batch_size)]
    inputs = convert_to_tensors(batch_input, model.device)
    input_ids_cutoff = inputs["input_ids"].size(dim=1)

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
    )

    return [format_output(out) for out in output]


def tokenize_opencode(tokenizer: PreTrainedTokenizer, prompt: str):
    input_ids = []
    attention_mask = []

    # verbose, but follows what is shown in the readme
    user = tokenizer("User:")
    prompt_text = tokenizer(f"""Create a Python script for this problem: {prompt}""")
    eot_token = tokenizer("<|end_of_turn|>")
    assistant = tokenizer("Assistant:")

    # verbose, but follows what is shown in the readme
    input_ids += user.input_ids
    input_ids += prompt_text.input_ids
    input_ids += eot_token.input_ids
    input_ids += assistant.input_ids

    # verbose, but follows what is shown in the readme
    attention_mask += user.attention_mask
    attention_mask += prompt_text.attention_mask
    attention_mask += eot_token.attention_mask
    attention_mask += assistant.attention_mask

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def convert_to_tensors(opencode_tokens: list[dict], device: torch.device):
    input_ids = [tokens["input_ids"] for tokens in opencode_tokens]
    attention_mask = [tokens["attention_mask"] for tokens in opencode_tokens]

    return {
        "input_ids": torch.tensor(input_ids).to(device),
        "attention_mask": torch.tensor(attention_mask).to(device),
    }


def run_eval(num_samples_per_task: int):
    problems = read_problems()

    tokenizer = AutoTokenizer.from_pretrained(
        "openchat/opencoderplus",
        use_auth_token=TOKEN,
    )

    model = torch.compile(
        GPTBigCodeForCausalLM.from_pretrained(
            "openchat/opencoderplus",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory={
                0: "18GiB",
                1: "18GiB",
            },
            use_auth_token=TOKEN,
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

    write_jsonl("results/opencode/eval.jsonl", samples)


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    os.makedirs("results/opencode", exist_ok=True)

    run_eval(num_samples_per_task)
