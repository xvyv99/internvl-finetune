import json
from pathlib import Path

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import dotenv_values

import tqdm

from utils import split_pic, cv2_to_base64, content_from_llm_block

config = dotenv_values(".env")

BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "opengvlab/internvl3-14b:free"
API_KEY = config['API_KEY']

model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY, # type: ignore
    base_url=BASE_URL,
)

img_segments = split_pic(Path('./data/test.jpg'), min_slice_height=4096)

b64_image = cv2_to_base64(img_segments[0], img_format='jpg')

with open('./export.md', 'w', encoding='utf-8') as f:
    for x in tqdm.tqdm(range(0, len(img_segments))):
        b64_image = cv2_to_base64(img_segments[x], img_format='jpg')

        messages = [
            SystemMessage("Convert the image into Markdown. Return a markdown code block, with no additional explanation."),
            HumanMessage(content=[{
                "type": "image",
                "source_type": "base64",
                "data": b64_image,
                "mime_type": "image/jpeg",
            }]),
        ]

        resp = model.invoke(messages)

        print(resp.content)

        assert isinstance(resp.content, str), f"Expected string, got {type(resp.content)}"
        markdown_content = content_from_llm_block(resp.content)

        f.write(markdown_content)
