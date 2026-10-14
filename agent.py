import asyncio
import json

import ollama

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


MODEL = "llama3.2:latest"


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # 1. Initialize the MCP connection
            await session.initialize()

            # 2. Discover the tools exposed by the server
            tools_response = await session.list_tools()

            tools = tools_response.tools

            print("Connected to MCP server.")
            print("Available tools:")

            for tool in tools:
                print("-", tool.name)

            # 3. Convert MCP tool schemas into Ollama tool definitions
            ollama_tools = []

            for tool in tools:
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema
                    }
                })

            messages = [{
                "role": "system",
                "content": (
                    """
                    You are a customer support assistant.

                    1. For troubleshooting questions, use search_knowledge_base and
                    answer using the retrieved information. Never invent policies
                    or troubleshooting instructions.

                    2. If the user wants to create a support ticket, collect the
                    required fields conversationally:
                    - subject
                    - description

                    Ask for missing fields one at a time. Do not ask the user
                    for their user ID.
                    Call create_support_ticket only when both fields are available.

                    3. Do not create a ticket unless the user requests one or agrees
                    to your offer.

                    4. For ticket status questions, always call get_ticket_status.
                    Never invent ticket information or claim an issue is resolved
                    without evidence.

                    5. If a tool fails, communicate the failure honestly.

                    6. Keep responses concise and professional.
                        """
                )
            }]

            while True:
                question = input("\nYou: ").strip()

                if question.lower() == "exit":
                    break

                if not question:
                    continue

                messages.append({
                    "role": "user",
                    "content": question
                })

                # 4. Let the LLM decide whether to call a tool
                for _ in range(5):
                    response = ollama.chat(
                        model=MODEL,
                        messages=messages,
                        tools=ollama_tools
                    )

                    assistant_message = response["message"]

                    messages.append(assistant_message)

                    tool_calls = assistant_message.get("tool_calls", [])

                    # No tool call means the LLM has generated its answer
                    if not tool_calls:
                        print("\nAssistant:", assistant_message["content"])
                        break

                    # 5. Execute each selected MCP tool
                    for tool_call in tool_calls:
                        function = tool_call["function"]
                        tool_name = function["name"]
                        arguments = function.get("arguments", {})

                        print(f"\nCalling tool: {tool_name}")

                        try:
                            result = await session.call_tool(
                                tool_name,
                                arguments=arguments
                            )
                            if tool_name == "get_ticket_status":
                                print("Tool result:", result.structuredContent)
                            print("Tool arguments:", arguments)
                            print("Tool result:", result)

                            result_text = "\n".join(
                                item.text
                                for item in result.content
                                if item.type == "text"
                            )

                            tool_output = {
                                "result": result_text,
                                "is_error": result.isError
                            }

                        except Exception as exc:
                            tool_output = {
                                "error": str(exc)
                            }

                        # 6. Return the tool result to the LLM
                        messages.append({
                            "role": "tool",
                            "content": json.dumps(tool_output),
                            "tool_name": tool_name
                        })

                else:
                    print(
                        "\nAssistant: I couldn't complete the request "
                        "within the tool-call limit."
                    )

                # Keep the conversation context manageable
                if len(messages) > 21:
                    messages = [messages[0]] + messages[-20:]


if __name__ == "__main__":
    asyncio.run(main())