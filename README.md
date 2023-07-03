# code-eval

## What

This is a repo I use to run human-eval on code models, adjust as needed. Some scripts adjusted from wizardcoder repo. The code is duplicated, mostly to handle edge cases around model tokenizing and loading (might eventually clean it up).

## Results
 
| model                                                                                         | size | pass@1 | pass@10 | screenshot                                                                                                         |
| --------------------------------------------------------------------------------------------- | ---- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------ |
| [WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)                  | 15B  | 57%    | 68.9%   | ![wizardcoder](https://github.com/abacaj/code-eval/assets/7272343/0b941ff8-b474-4236-bbc0-89d925bbd34e)            |
| [openchat/opencoderplus](https://huggingface.co/openchat/opencoderplus)                       | 15B  | N/A    | N/A     | pending                                                                                                            |
| [teknium/Replit-v1-CodeInstruct-3B](https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B) | 3B   | 25.8%  | 42.6%   | ![replit-codeinstruct-v1](https://github.com/abacaj/code-eval/assets/7272343/4fca98d8-2c22-43ce-9639-e998ecb4fedc) |
| [teknium/Replit-v2-CodeInstruct-3B](https://huggingface.co/teknium/Replit-v2-CodeInstruct-3B) | 3B   | 21.5%  | 31%     | ![replit-codeinstruct-v2](https://github.com/abacaj/code-eval/assets/7272343/655aaa1d-0715-4fcd-b9ba-a22b5fddb215) |
| [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b)                          | 3B   | 15.1%  | 27.4%   | ![replit-code-v1](https://github.com/abacaj/code-eval/assets/7272343/53375b9e-9054-4e8d-936a-1b1e7d13c291)         |


## Setup

Create python environment

```sh
python -m venv env && source env/bin/activate
```

Install dependencies

```sh
pip install -r requirements.txt
```

Run the eval script

```sh
# replace script file name for various models:
# eval_wizard.py
# eval_opencode.py
# eval_replit.py

python eval_wizard.py
```

Process the jsonl file to extract code samples from model completions

**Note**: the replit base + instruct model does not go through this process

```sh
# replace args for various models:
# --path results/wizard --out_path results/wizard/eval.jsonl
# --path results/opencode --out_path results/opencode/eval.jsonl

python process_eval.py --path results/wizard --out_path results/wizard/processed.jsonl --add_prompt
```

Then get the results

```sh
# replace args for various models:
# results/wizard/processed.jsonl
# results/opencode/processed.jsonl
# results/replit_instruct/eval.jsonl
# results/replit/eval.jsonl

evaluate_functional_correctness results/wizard/processed.jsonl
```
