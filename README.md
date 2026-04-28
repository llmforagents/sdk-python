# llm4agents-sdk

Python SDK for [llm4agents.com](https://llm4agents.com) — gasless AI agent infrastructure.

## Install

pip install llm4agents-sdk

## Quick start

```python
import asyncio
from llm4agents import LLM4AgentsClient

async def main():
    client = LLM4AgentsClient(api_key="sk-proxy-...")
    conv = client.chat.conversation({"model": "openai/gpt-4o"})
    result = await conv.say("Hello!")
    print(result.content)

asyncio.run(main())
```

## Version

This SDK tracks the TypeScript SDK. Version parity is maintained via CONTRACT.md in the TS SDK repo.
