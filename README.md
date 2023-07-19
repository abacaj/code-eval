# code-eval

## What

This is a repo I use to run human-eval on code models, adjust as needed. Some scripts were adjusted from wizardcoder repo ([process_eval.py](https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/process_humaneval.py)). The evaluation code is duplicated in several files, mostly to handle edge cases around model tokenizing and loading (will clean it up).

## Results

Table is sorted by pass@1 score.
 
| model                                                                                                 | size | pass@1  | pass@10 | screenshot                                                                                                         |
| ----------------------------------------------------------------------------------------------------- | ---- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| [sahil2801/replit-code-instruct-glaive](https://huggingface.co/sahil2801/replit-code-instruct-glaive) | 3B   | 63.5%   | 67%     | ![instruct-glaive](https://github.com/abacaj/code-eval/assets/7272343/6fd7527d-0dc4-4b48-8a57-ad0373074bc5)        |
| [WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)                          | 15B  | 57%     | 68.9%   | ![wizardcoder](https://github.com/abacaj/code-eval/assets/7272343/0b941ff8-b474-4236-bbc0-89d925bbd34e)            |
| [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)                                         | 15B  | 34.6%   | 48.7%   | ![starcoder](https://github.com/abacaj/code-eval/assets/7272343/eb5df978-f56b-4557-a433-8b8fa863a059)              |
| [openchat/opencoderplus](https://huggingface.co/openchat/opencoderplus)                               | 15B  | 27.3%   | 43.9%   | ![opencoder](https://github.com/abacaj/code-eval/assets/7272343/1fa9f5ef-941b-4ea8-981e-c3f258c03fee)              |
| [teknium/Replit-v1-CodeInstruct-3B](https://huggingface.co/teknium/Replit-v1-CodeInstruct-3B)         | 3B   | 25.8%   | 42.6%   | ![replit-codeinstruct-v1](https://github.com/abacaj/code-eval/assets/7272343/4fca98d8-2c22-43ce-9639-e998ecb4fedc) |
| [teknium/Replit-v2-CodeInstruct-3B](https://huggingface.co/teknium/Replit-v2-CodeInstruct-3B)         | 3B   | 21.5%   | 31%     | ![replit-codeinstruct-v2](https://github.com/abacaj/code-eval/assets/7272343/655aaa1d-0715-4fcd-b9ba-a22b5fddb215) |
| [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b)                                  | 3B   | 17.1%   | 29.8%   | ![replit-code-v1](https://github.com/abacaj/code-eval/assets/7272343/6b387aa8-db60-4f04-b458-35b010b1145c)         |
| [mpt-7b](https://huggingface.co/mosaicml/mpt-7b)                                                      | 7B   | 15.9%   | 23.7%   | ![mpt-7b](https://github.com/abacaj/code-eval/assets/7272343/16965905-a368-4254-aeab-5e44126eba84)                 |
| [xgen-7b-8k-base](https://huggingface.co/Salesforce/xgen-7b-8k-base)                                  | 7B   | 14.9%   | 22.5%   | ![xgen-7b-8k-base](https://github.com/abacaj/code-eval/assets/7272343/995c84a9-ee69-43bf-8502-a74eba1d927a)        |
| [openllama-7b-v2](https://huggingface.co/openlm-research/open_llama_7b)                               | 7B   | 14%     | 23.1%   | ![openllama-7b-v2](https://github.com/abacaj/code-eval/assets/7272343/e38f08a0-ae74-4c51-b3a7-638781477e1b)        |
| [llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)                                                | 7B   | 13.1%   | 21.9%   | ![llama-2-7b](https://github.com/abacaj/code-eval/assets/7272343/cc86cc7c-beac-4993-9ca3-d91a48a790e4)                                                                                               |
| [llama-7b](https://huggingface.co/huggyllama/llama-7b)                                                | 7B   | 12.1%   | 18.9%   | ![llama-7b](https://github.com/abacaj/code-eval/assets/7272343/605a3c4e-0b2b-4c10-a185-f2a4d34ec10d)                                                                                               |
| [mpt-30b](https://huggingface.co/mosaicml/mpt-30b)                                                    | 30B  | pending | pending | pending                                                                                                            |

## FAQ

> Why is there a discrepancy on some of the scores between official numbers? 

Because it is not obvious or published what prompt or processing the official models used to conduct their evaluation on this benchmark. The goal here is to try and best reproduce those numbers, in many cases it is possible to get very close to the published numbers.

All of the scores here were run independently of any published numbers and are reproducible by cloning the repo and following the setup.

> Why do some models have a filter_code post generation step?

Base models can in many cases repeat outputs, breaking the benchmark scores. Instruct models don't have this problem and so you won't see this step, they tend to output a end of sequence token.

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

Process the jsonl file to extract code samples from model completions.

**Note**: Only wizard & opencoder require this, they return markdown output with code.

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
