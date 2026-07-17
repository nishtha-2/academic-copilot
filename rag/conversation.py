conversation = []


def add_user_message(message):

    conversation.append({
        "role": "user",
        "content": message
    })


def add_assistant_message(message):

    conversation.append({
        "role": "assistant",
        "content": message
    })


def get_history(limit=6):

    return conversation[-limit:]


def clear_history():

    conversation.clear()