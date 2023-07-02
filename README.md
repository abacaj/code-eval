# code-eval

## What

This is a repo I use to run human-eval on code models, adjust as needed. Adjusted from wizardcoder repo.

## Results

#### [WizardCoder](WizardLM/WizardCoder-15B-V1.0) 
| pass@1 | pass@10 |
| ------ | ------- |
| 57%    | 68.9%   |

![wizardcoder](https://github.com/abacaj/code-eval/assets/7272343/0b941ff8-b474-4236-bbc0-89d925bbd34e)

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
# adjust file name for various models:
# eval_wizard.py
# eval_opencode.py
# eval_replit.py

python eval_wizard.py
```

Process the jsonl file to extract code samples from model completions

```sh
# adjust args for various models:
# --path wizard_eval --out_path wizard_eval.jsonl
# --path opencode_eval --out_path opencode_eval.jsonl
# --path replit_eval --out_path replit_eval.jsonl

python process_eval.py --path wizard_eval --out_path wizard_eval.jsonl --add_prompt
```

Then get the results

```sh
# adjust file for various models:
# wizard_eval.jsonl
# opencode_eval.jsonl
# replit_eval.jsonl

evaluate_functional_correctness wizard_eval.jsonl
```
