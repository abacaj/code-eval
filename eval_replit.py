from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from core import run_eval
import os
import torch

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    prompt_input = f"""Complete the following Python code without any tests or explanation\n{prompt}"""

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

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    out_path = "results/replit/eval.jsonl"
    os.makedirs("results/replit", exist_ok=True)

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

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
    )
