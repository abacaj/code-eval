# code-eval

## What

This is a repo I use to run human-eval on code models, adjust as needed. Adjusted from wizardcoder repo.

## Results

#### [WizardCoder](WizardLM/WizardCoder-15B-V1.0) 
| pass@1 | pass@10 |
| --------- | ------------ |
| 57%       | 68.9%        |

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

Run wizardcoder

```sh
python eval_wizard.py
```

Run opencoder

```sh
python eval_opencode.py
```

Process the jsonl file using wizardcoder script to extract the code samples

```sh
python process_eval.py --path wizard_eval --out_path wizard_eval.jsonl --add_prompt
```

Or for opencode

```sh
python process_eval.py --path opencode_eval --out_path opencode_eval.jsonl --add_prompt
```

Then run the eval script

```sh
evaluate_functional_correctness wizard_eval.jsonl
```

Or

```sh
evaluate_functional_correctness opencode_eval.jsonl
```
