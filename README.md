# code-eval

## What

This is a repo I use to run human-eval on code models, adjust as needed. Some scripts were adjusted from wizardcoder repo (`process_eval.py`). The evaluation code is duplicated in several files, mostly to handle edge cases around model tokenizing and loading (might eventually clean it up).

## Results

Table is sorted by pass@1 score.
 
| model                                                                                                 | size | pass@1 | pass@10 | screenshot                                                                                                         |
| ----------------------------------------------------------------------------------------------------- | ---- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------ |
| [sahil2801/replit-code-instruct-glaive](https://huggingface.co/sahil2801/replit-code-instruct-glaive) | 3B   | 63.5%  | 67%     | ![instruct-glaive](https://github.com/abacaj/code-eval/assets/7272343/6fd7527d-0dc4-4b48-8a57-ad0373074bc5)        |
| [WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)                          | 15B  | 57%    | 68.9%   | ![wizardcoder](https://github.com/abacaj/code-eval/assets/7272343/0b941ff8-b474-4236-bbc0-89d925bbd34e)            |
| [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)                                         | 15B  | 34.6%  | 48.7%   | ![starcoder](https://github.com/abacaj/code-eval/assets/7272343/eb5df978-f56b-4557-a433-8b8fa863a059)              |
| [openchat/opencoderplus](https://huggingface.co/openchat/opencoderplus)                               | 15B  | 27.3%  | 43.9%   | ![opencoder](https://github.com/abacaj/code-eval/assets/7272343/1fa9f5ef-941b-4ea8-981e-c3f258c03fee)              |
| [teknium/Replit-v1-CodeInstruct-3B](https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B)         | 3B   | 25.8%  | 42.6%   | ![replit-codeinstruct-v1](https://github.com/abacaj/code-eval/assets/7272343/4fca98d8-2c22-43ce-9639-e998ecb4fedc) |
| [teknium/Replit-v2-CodeInstruct-3B](https://huggingface.co/teknium/Replit-v2-CodeInstruct-3B)         | 3B   | 21.5%  | 31%     | ![replit-codeinstruct-v2](https://github.com/abacaj/code-eval/assets/7272343/655aaa1d-0715-4fcd-b9ba-a22b5fddb215) |
| [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b)                                  | 3B   | 15.1%  | 27.4%   | ![replit-code-v1](https://github.com/abacaj/code-eval/assets/7272343/53375b9e-9054-4e8d-936a-1b1e7d13c291)         |
| [xgen-7b-8k-base](https://huggingface.co/Salesforce/xgen-7b-8k-base)                                  | 7B   | 14.6%  | 20.7%   | ![xgen-7b-8k-base](https://github.com/abacaj/code-eval/assets/7272343/b2388ea1-382d-4bfe-9f08-68227e8cf8ad)        |
| [mpt-30b](https://huggingface.co/mosaicml/mpt-30b)                                                    | 30B  | 14.4%  | 31.7%   | ![mpt-30b](https://github.com/abacaj/code-eval/assets/7272343/f0a082f9-35bc-423c-ad50-c21d90f2447c)                |
| [mpt-7b](https://huggingface.co/mosaicml/mpt-7b)                                                      | 7B   | 11.7%  | 14%     | ![mpt-7b](https://github.com/abacaj/code-eval/assets/7272343/f0a082f9-35bc-423c-ad50-c21d90f2447c)                 |

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
# eval_mpt.py
# eval_starcoder.py
# eval_replit.py
# eval_replit_glaive.py
# eval_replit_instruct.py

python eval_wizard.py
```

Process the jsonl file to extract code samples from model completions

**Note**: the replit base, instruct model, and starcoder does not go through this process

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
# results/starcoder/eval.jsonl
# results/mpt/eval.jsonl
# results/opencode/processed.jsonl
# results/replit_instruct/eval.jsonl
# results/replit_glaive/eval.jsonl
# results/replit/eval.jsonl

evaluate_functional_correctness results/wizard/processed.jsonl
```
