import os
import base64
from pathlib import Path

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import dotenv_values

from utils import split_pic, cv2_to_base64

config = dotenv_values(".env")

BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "opengvlab/internvl3-14b:free"
API_KEY = config['API_KEY']

model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY, # type: ignore
    base_url=BASE_URL,
)

with open("test1.jpg", "rb") as image_file:
    b64_image = base64.b64encode(image_file.read()).decode("utf-8")

img_segments = split_pic(Path('./test.jpg'), write_file=True)

b64_image = cv2_to_base64(img_segments[0], img_format='jpg')

messages = [
    SystemMessage("Please convert the following picture to Markdown format. Just return the Markdown code block."),
    HumanMessage(content=[{
        "type": "image",
        "source_type": "base64",
        "data": b64_image,
        "mime_type": "image/jpeg",
    }]),
]

resp = model.invoke(messages)

print(resp.content)
