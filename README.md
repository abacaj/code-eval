# code-eval

## What

This is a repo I use to run human-eval on code models, adjust as needed. Some scripts adjusted from wizardcoder repo. The code is duplicated, mostly to handle edge cases around model tokenizing and loading (might eventually clean it up).

## Results
 
| model                                                               | pass@1 | pass@10 | screenshot                                                                                              |
| ------------------------------------------------------------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------- |
| [WizardCoder](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0) | 57%    | 68.9%   | ![wizardcoder](https://github.com/abacaj/code-eval/assets/7272343/0b941ff8-b474-4236-bbc0-89d925bbd34e) |



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
**Note**: replit base model does not go through this process

```sh
# replace args for various models:
# --path results/wizard --out_path results/wizard/eval.jsonl
# --path results/opencode --out_path results/opencode/eval.jsonl
# --path results/replit_instruct --out_path results/replit_instruct/eval.jsonl

python process_eval.py --path results/wizard --out_path results/wizard/processed.jsonl --add_prompt
```

Then get the results

```sh
# replace args for various models:
# results/wizard/processed.jsonl
# results/opencode/processed.jsonl
# results/replit_instruct/processed.jsonl
# results/replit/eval.jsonl

evaluate_functional_correctness results/wizard/processed.jsonl
```
