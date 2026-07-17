from ollama import chat


def generate_followups(query, answer):
    """
    Generate intelligent follow-up questions based on the
    user's query and the assistant's answer.
    """

    prompt = f"""
You are an expert academic tutor.

The student asked:

{query}

You answered:

{answer}

Now generate 4 intelligent follow-up questions.

Rules:

- Questions should deepen understanding.
- Don't repeat the original question.
- Make them short.
- They should naturally continue the conversation.
- Include at least one practical/application question.
- Include at least one exam/interview style question.

Return ONLY the questions.

Example format:

1. ...
2. ...
3. ...
4. ...
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