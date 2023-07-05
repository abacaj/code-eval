from transformers import (
    AutoTokenizer,
    GPTBigCodeForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from core import run_eval, standard_prompt
import os
import torch

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int
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
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return batch_completions


def tokenize_opencode(tokenizer: PreTrainedTokenizer, prompt: str):
    input_ids = []
    attention_mask = []

    # verbose, but follows what is shown in the readme
    user = tokenizer("User:")
    prompt_text = tokenizer(standard_prompt(prompt))
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


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    out_path = "results/opencode/eval.jsonl"
    os.makedirs("results/opencode", exist_ok=True)

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

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )
