from datetime import datetime
from pathlib import Path
from enum import Enum

import requests
from dotenv import dotenv_values

config = dotenv_values(".env")
ELSEVIER_API_KEY = config["ELSEVIER_API_KEY"]

class ArticleFormat(Enum):
    PDF = ('application/pdf', 'pdf')
    IMG = ('image/png', 'png')
    TXT = ('text/plain', 'txt')

def get_article_by_doi(doi: str, format_type: ArticleFormat=ArticleFormat.PDF) -> Path:

    BASE_URL = f"https://api.elsevier.com/content/article/doi/{doi}"
    headers = {
        "X-ELS-APIKey": ELSEVIER_API_KEY,
        "Accept": format_type.value[0]
    }

    file_path = Path('./') / (doi.replace("/", "_").replace(".", "_") + '.' + format_type.value[1])

    response = requests.get(BASE_URL, headers=headers, stream=True)
    response.raise_for_status()

    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return file_path

def speed_test(download_times: int) -> None:
    time_lst = []
    size_lst = []

    try:
        for i in range(download_times):
            start_time_per = datetime.now()
            file_path = get_article_by_doi("10.1016/j.eswa.2025.127661")
            end_time_per = datetime.now()

            file_size = file_path.stat().st_size
            time_usage = end_time_per - start_time_per

            size_lst.append(file_size)
            time_lst.append(time_usage.seconds)

            print(f"Range {i}: ", time_usage)
    except Exception as e:
        raise e
    finally:
        print("Speed: ", sum(size_lst)/sum(time_lst), "b/s")

if __name__ == '__main__':
    speed_test(3)
