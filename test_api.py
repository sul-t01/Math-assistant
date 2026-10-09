from dotenv import load_dotenv
load_dotenv()

import asyncio
from llm import ask_llm

print(asyncio.run(ask_llm("math", "What is 12 * 13? Answer briefly.")))