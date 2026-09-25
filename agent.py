
import ollama

from tools import (
    list_files,
    read_file,
    write_file,
    create_directory,
)


# ============================================================
# TOOL DEFINITIONS
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories inside the workspace. "
                "Use this to observe the current workspace state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory path relative to the workspace. "
                            "Use '.' for the workspace root."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a text file inside the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a text file inside the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": (
                "Create a directory inside the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the workspace.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


# ============================================================
# PYTHON FUNCTIONS THAT ACTUALLY EXECUTE THE TOOLS
# ============================================================

tool_functions = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "create_directory": create_directory,
}


# ============================================================
# AGENT
# ============================================================

def run_agent(user_message: str):

    messages = [
        {
            "role": "system",
            "content": """
You are an autonomous Linux workspace agent.

Your job is to accomplish the user's task by interacting
with the workspace through the available tools.

You operate using this loop:

1. OBSERVE
   - Inspect the current state of the workspace.
   - Use observation tools such as list_files or read_file.
   - Do not assume that files or directories exist.

2. DECIDE
   - Analyze the user's goal and the information you observed.
   - Decide what action should happen next.
   - Choose the appropriate tool and arguments.
   - For complex tasks, break the task into smaller steps.

3. ACTUATE
   - Execute the chosen action using the appropriate tool.
   - Never claim that an action happened unless the tool
     actually returned a successful result.

4. RECHECK
   - After every meaningful modification, observe the
     workspace again.
   - Verify that the intended change actually happened.
   - If the result is incorrect or incomplete, decide on
     another action and continue.

The fundamental operating loop is:

OBSERVE → DECIDE → ACTUATE → RECHECK

Repeat this loop until the user's task has been completed
and verified.

Rules:

- Always use tools when filesystem information is required.
- Always use tools when filesystem changes are requested.
- Do not assume the state of the workspace.
- Verify important changes after making them.
- Never claim success without verification.
- Stay inside the provided workspace.
- Do not perform destructive operations unless explicitly requested.
- Do not invent tool results.
- If a tool returns an error, use that information to decide
  what to do next.
""",
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    # ========================================================
    # AGENT LOOP
    # ========================================================

    while True:

        print("\n[THINKING]")

        response = ollama.chat(
            model="qwen3:4b",
            messages=messages,
            tools=tools,
        )

        message = response["message"]

        # Add the assistant's response to the conversation.
        messages.append(message)

        # ----------------------------------------------------
        # If there are no tool calls, the model has finished.
        # ----------------------------------------------------

        if not message.get("tool_calls"):
            return message.get("content", "")

        # ----------------------------------------------------
        # Execute the requested tools.
        # ----------------------------------------------------

        for call in message["tool_calls"]:

            function_name = call["function"]["name"]
            arguments = call["function"]["arguments"]

            print(f"\n[ACTUATE] {function_name}")
            print(f"[ARGS] {arguments}")

            function = tool_functions.get(function_name)

            # ------------------------------------------------
            # Unknown tool
            # ------------------------------------------------

            if function is None:

                result = (
                    f"Error: unknown tool '{function_name}'."
                )

            # ------------------------------------------------
            # Execute tool
            # ------------------------------------------------

            else:

                try:
                    result = function(**arguments)

                except Exception as error:
                    result = (
                        f"Tool execution failed: {error}"
                    )

            print(f"[RESULT] {result}")

            # ------------------------------------------------
            # Give the tool result back to the model.
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "content": str(result),
                }
            )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("====================================")
    print("      LOCAL AI WORKSPACE AGENT")
    print("====================================")
    print("Model: qwen3:4b")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:

        try:
            user_input = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            answer = run_agent(user_input)

            print("\nAgent:")
            print(answer)

        except Exception as error:
            print("\n[ERROR]")
            print(error)

