from ollama import chat


def explain_code(code):

    prompt = f"""
You are an expert programming tutor.

Explain the following code line by line.

For every line:
- Explain what it does.
- Explain why it is used.
- Mention any mistakes if present.

Code:
{code}
"""

    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def debug_code(code):

    prompt = f"""
You are an expert debugger.

Find every bug in the following code.

For each bug:
1. Explain the problem.
2. Explain why it occurs.
3. Give the corrected code.

Code:
{code}
"""

    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def generate_program(question):

    prompt = f"""
You are an expert programmer.

Write a complete solution for the following problem.

Requirements:
- Correct code
- Well commented
- Efficient
- Explain the approach after the code

Problem:
{question}
"""

    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]