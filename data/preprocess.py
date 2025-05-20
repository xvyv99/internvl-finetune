from dotenv import dotenv_values

config = dotenv_values(".env")

BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "opengvlab/internvl3-14b:free"
API_KEY = config['API_KEY']

from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = ChatOpenAI(
    api_key=API_KEY, # type: ignore
    base_url=BASE_URL,
    model=MODEL_NAME,
)

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

print(llm.invoke(messages))
