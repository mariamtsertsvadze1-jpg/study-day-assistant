import os
import getpass
from langchain_core.messages import ToolMessage
from agent import build_graph

def main():
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your GROQ_API_KEY: ")

    graph = build_graph()
    config = {"configurable": {"thread_id": "cli-session-1"}}

    print("\n=== Study Day Assistant (CLI) ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        try:
            result = graph.invoke({"messages": [("user", user_input)]}, config)
            state = graph.get_state(config)

            if state.next:
                tool_call = state.values["messages"][-1].tool_calls[0]
                task_name = tool_call["args"].get("task", "")
                confirm = input(f"⚠️ Confirm deleting task '{task_name}'? (yes/no): ").strip().lower()

                if confirm == "yes":
                    resumed = graph.invoke(None, config)
                    print(f"Assistant: {resumed['messages'][-1].content}\n")
                else:
                    cancel_msg = ToolMessage(
                        tool_call_id=tool_call["id"],
                        content="Action cancelled by user. The task was NOT deleted."
                    )
                    graph.update_state(config, {"messages": [cancel_msg]}, as_node="sensitive_tools")
                    resumed = graph.invoke(None, config)
                    print(f"Assistant: {resumed['messages'][-1].content}\n")
            else:
                print(f"Assistant: {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
