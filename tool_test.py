import ollama


def create_directory(path: str):
    """Create a directory in the workspace."""
    print(f"TOOL CALLED: create_directory({path})")
    return f"Directory created: {path}"


response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "Create a directory called test."
        }
    ],
    tools=[create_directory],
)

print(response["message"])

if response["message"].get("tool_calls"):
    for call in response["message"]["tool_calls"]:
        print("Tool:", call["function"]["name"])
        print("Arguments:", call["function"]["arguments"])KO