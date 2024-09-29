from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import os
from core.funs import num_tokens_from_string
from dotenv import load_dotenv

load_dotenv()
chain = (PromptTemplate(template="""{query}""",
                        input_variables=["query"])
         | ChatOpenAI(model=os.getenv('DEFAULT_LARGE_MODEL'), temperature=0))
embeddings = OpenAIEmbeddings(model=os.getenv('DEFAULT_EMBEDDING_MODEL'))


def qa(query: str) -> str:
    # print(query)
    return chain.invoke({'query': query}).content


def embed(query: str) -> list: return embeddings.embed_query(query)


from DTS import *


def get_data() -> list[str]:
    data = []
    for i in [0, 100, 200, 300]:
        with open('doc2dial/dialogue_' + str(i) + '.txt', 'r') as f:
            data_n = f.readlines()
            data.extend(data_n)

    for i in range(len(data) - 1, -1, -1):
        data[i] = data[i].replace('\n', '')
        if data[i] == "================": data.pop(i)
    return data


def get_data_2(df) -> list[str]:
    # df = pd.read_csv('messages_test.csv')
    keys = df.keys()
    data = []

    for i in range(len(df[keys[0]])):
        StrContent = df['StrContent'][i]
        StrTime = df['StrTime'][i]
        Sender = df['Sender'][i]

        if StrContent != StrContent: StrContent = '[文件消息]'
        if "<VoIPBubbleMsg>" in StrContent: StrContent = '[语音消息]'
        if "<img" in StrContent: StrContent = '[图片消息]'
        if "bigheadimgurl" in StrContent: StrContent = '[用户推荐消息]'
        if "<location" in StrContent: StrContent = '[定位消息]'
        if "<videomsg" in StrContent: StrContent = '[视频消息]'
        if "<revokemsg>" in StrContent: continue

        data.append(f"[{StrTime}]{Sender}: {StrContent}")
    return data


def process_wechat(data):
    # 数据获取
    dialo = get_data_2(data)
    # 对话摘要
    DSer = Dialo_summarizer(dialo_new=dialo,  # 对话数据,仅读取结果文档时取值为[]
                            file_path="test.csv",  # 存储对话数据的路径
                            init_flag=True)  # 是否初始化，False：基于文件中已有数据进行增量摘要
    return DSer.run()
    # DSer.show()
    # DSer.evluate()

