from dotenv import load_dotenv
import nest_asyncio
from llama_index.llms.openai import OpenAI


load_dotenv()

nest_asyncio.apply()

llm = OpenAI(model='gpt-4o-mini')




