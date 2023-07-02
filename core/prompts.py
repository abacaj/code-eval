def instruct_prompt(prompt: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation\n{prompt}\n\n### Response:"""


def standard_prompt(prompt: str) -> str:
    return f"""Complete the following Python code without any tests or explanation\n{prompt}"""
