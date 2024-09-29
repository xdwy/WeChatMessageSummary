import re
import tiktoken
import pandas as pd


def jsonExtract(text):
    pattern = re.compile(r'```json(.*?)```', re.DOTALL)
    res = pattern.findall(text)[0]
    return res


def bashExtract(text):
    pattern = re.compile(r'```bash(.*?)```', re.DOTALL)
    res = pattern.findall(text)
    return res




def append2csv(data, filePath, columns=['info']):
    df_n = pd.DataFrame(data, columns=columns)
    try:
        df = pd.read_csv(filePath)
    except:
        df_temp = pd.DataFrame([], columns=columns)
        df_temp.to_csv(filePath, index=False)
        df = pd.read_csv(filePath)
    df = df._append(df_n)
    df.to_csv(filePath, index=False)


import ast


def csvInfoCheck(csv_path):
    df = pd.read_csv(csv_path)
    for item in df['info']:
        try:
            temp_content = ast.literal_eval(item)
            print(temp_content)
        except:
            return False

    checked_csv_path = csv_path.replace('unchecked', 'checked')
    df.to_csv(checked_csv_path)
    return True

def num_tokens_from_string(string: str, encoding_name: str = 'p50k_base') -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
