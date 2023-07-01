# code-val

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