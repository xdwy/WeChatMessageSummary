import copy
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import os
import jieba
import pandas as pd
from rouge_chinese import Rouge
import logging
logging.basicConfig(level=logging.INFO)
from core.funs import num_tokens_from_string
from dotenv import load_dotenv
load_dotenv()
chain = (PromptTemplate(template="""{query}""",
                        input_variables=["query"])
         | ChatOpenAI(model=os.getenv('DEFAULT_LARGE_MODEL'), temperature=0.5))
embeddings = OpenAIEmbeddings(model = os.getenv('DEFAULT_EMBEDDING_MODEL'))

from Prompts import *
def qa(query:str) -> str:
    # print(query)
    return chain.invoke({'query': query}).content

def get_rouge(hypothesis, reference):
    hypothesis = ' '.join(jieba.cut(hypothesis))
    reference = ' '.join(jieba.cut(reference))
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores
class Fragment():
    dialo: list[str] = []
    seg: list[int] = []
    summary: str = 'nothing'

    def __init__(self,
                 dialo: list[str],
                 seg: list[int],
                 summary: str):
        self.dialo = dialo
        self.seg = seg
        self.summary = summary

    def extend(self, frag_B):
        self.dialo.extend(frag_B.dialo)
        self.seg.extend(frag_B.seg)

        res = sorted(self.seg)
        self.dialo = [self.dialo[self.seg.index(item)] for item in res]
        self.seg.sort()
        self.summary = Summary(self.dialo)
class Dialo_summarizer():
    dialo: list[str] = []
    Seg: list[list] = []
    Summary: list[str] = []

    def __init__(self,
                 dialo_new: list,
                 file_path: str = "test.csv",
                 init_flag: bool = True):
        self.dialo = dialo_new
        self.file_path = file_path
        self.Summary_path = self.file_path.replace('.csv', '_summary.csv')

        if not init_flag: self.reload()
        # self.save()
        self.summarized_size = self.getNumberSummarized()

    def getNumberSummarized(self):
        try:
            return max([max(item) for item in self.Seg])+1
        except:
            return 0

    def run(self):
        # Split
        logging.info("正在进行对话片段分割……")
        Seg_res = Split(self.dialo, self.Seg, self.summarized_size)

        # Fragment segmentation
        logging.info("正在进行对话片段汇总……")
        frag_Seg_res = Fragment_segmentation(self.dialo, Seg_res)

        # Fragment integration
        logging.info("正在进行对话主题摘要……")
        Fragments = Fragment_integration(self.dialo, frag_Seg_res)

        self.Seg = []
        self.Summary = []
        for item in Fragments:
            self.Seg.append(item.seg)
            self.Summary.append(item.summary)
        data = [[self.Seg[i], self.Summary[i]] for i in range(len(self.Seg))]
        df = pd.DataFrame(data=data, columns=['Segmentation', 'Summary'])
        return df
        # self.save()

    def show(self):
        Summary, Seg = self.Summary, self.Seg
        for i in range(len(Seg)):
            content = ""
            content = '\n\t'.join([self.dialo[j] for j in Seg[i]])
            summary = Summary[i]
            print(Answer_Template.format(Serial_Num=i + 1,
                                         summary=summary,
                                         content=content))
    def evluate(self):
        Summary, Seg = self.Summary, self.Seg
        scores = []
        for i in range(len(Seg)):
            hypothesis = Summary[i]
            reference = '\n\t'.join([self.dialo[j] for j in Seg[i]])
            score = get_rouge(hypothesis, reference)
            scores.append(score)
        res = [[] for i in range(9)]
        for item in scores:
            res[0].append(item[0]['rouge-1']['r'])
            res[1].append(item[0]['rouge-1']['p'])
            res[2].append(item[0]['rouge-1']['f'])
            res[3].append(item[0]['rouge-2']['r'])
            res[4].append(item[0]['rouge-2']['p'])
            res[5].append(item[0]['rouge-2']['f'])
            res[6].append(item[0]['rouge-l']['r'])
            res[7].append(item[0]['rouge-l']['p'])
            res[8].append(item[0]['rouge-l']['f'])

        res = [round(sum(item)/len(item),3) for item in res]
        print(f"rouge-1:\n\tr:{res[0]}\n\tp:{res[1]}\n\tf:{res[2]}\nrouge-2:\n\tr:{res[3]}\n\tp:{res[4]}\n\tf:{res[5]}\nrouge-l:\n\tr:{res[6]}\n\tp:{res[7]}\n\tf:{res[8]}\n")
    def reload(self):
        df = pd.DataFrame(data=[], columns=['dialos'])
        try:
            df = pd.read_csv(self.file_path)
            df_summary = pd.read_csv(self.Summary_path)
            self.Seg = list(df_summary['Segmentation'])
            self.Seg = [eval(item) for item in self.Seg]
            self.Summary = list(df_summary['Summary'])
        except: pass
        dialo_old = list(df['dialos'])
        dialo_old.extend(self.dialo)
        self.dialo = dialo_old[:]

    def save(self):
        df = pd.DataFrame(data=self.dialo, columns=['dialos'])
        df.to_csv(self.file_path, index=False)
        logging.info(f"对话文件存储于：{self.file_path}")
        data = [[self.Seg[i], self.Summary[i]] for i in range(len(self.Seg))]
        df = pd.DataFrame(data=data, columns=['Segmentation', 'Summary'])
        df.to_csv(self.Summary_path, index=False)
        logging.info(f"摘要文件存储于：{self.Summary_path}")

def Split(dialo: list, Seg: list[list], pos_now: int) -> list[list]:
    window_size = 15
    logging.info(f"分割进度：{round(pos_now / len(dialo) * 100, 3)}%")
    while pos_now < len(dialo):
        pos_new = pos_now + min([window_size, len(dialo) - pos_now])
        dialo_now = [dialo[i] for i in range(pos_now, pos_new)]

        Seg_res_now = DTS(dialo_now)
        Seg_res_now = [[sub_item + pos_now for sub_item in item] for item in Seg_res_now]

        if len(Seg):
            tail_dialo = [dialo[i] for i in Seg[-1]]
            head_dialo = [dialo[i] for i in Seg_res_now[0]]
            if TS(tail_dialo, head_dialo):
                Seg[-1].extend(Seg_res_now[0])
                Seg_res_now.pop(0)

        Seg.extend(Seg_res_now)
        pos_now = copy.copy(pos_new)
        logging.info(f"分割进度：{round(pos_now / len(dialo) * 100, 3)}%")

    return Seg

def Fragment_segmentation(dialo: list, Seg_res:list[list]) -> list[list]:
    frag_window_size = 5
    frag_Seg_res = []
    pos_now = 0
    while pos_now < len(Seg_res):
        pos_new = pos_now + min([frag_window_size, len(Seg_res) - pos_now])
        farg_Seg_now = [Seg_res[i] for i in range(pos_now, pos_new)]

        farg_Seg_now = Seg_check(dialo, farg_Seg_now)
        frag_Seg_res.extend(farg_Seg_now)

        pos_now = copy.copy(pos_new)

    return frag_Seg_res
def Fragment_integration(dialo: list, frag_Seg_res:list[list]) -> list[Fragment]:

    dialo_set = [[dialo[i] for i in item] for item in frag_Seg_res]
    Fragments = []
    ## Summary
    pos_now = 0
    logging.info(f"摘要进度：{round(pos_now / len(frag_Seg_res) * 100, 3)}%")
    for i in range(len(dialo_set)):
        Fragments.append(Fragment(dialo_set[i], frag_Seg_res[i], Summary(dialo_set[i])))
        pos_now += 1
        logging.info(f"摘要进度：{round(pos_now / len(frag_Seg_res) * 100, 3)}%")

    ## Integration
    logging.info("正在进行对话主题汇总……")
    Fragments = Seg_summary_check(Fragments)
    logging.info("对话主题汇总完成")

    return Fragments
def DTS(dialo: list) -> list[list]:

    for i in range(len(dialo)):
        dialo[i] = f"{str(i)}-{dialo[i]}"
    Number_list = [t for t in range(i+1)]
    flag = True
    Seg_res = []
    while flag:
        try:
            res = qa(DTS_Template.format(dialo = '\n'.join(dialo),
                                         Number_list = Number_list))
            res = res.replace('\n', ',')
            Seg_res = eval(f'[{res}]')
            flag = False
        except: flag = True

    try: [[j for j in k] for k in Seg_res]
    except:
        temp = []
        for item in Seg_res:
            if type(item) == int:
                temp.append([item])
            elif type(item) == list:
                temp.append(item)
        Seg_res = temp[:]
    return Seg_res

def TS(dialo_A: list, dialo_B: list) -> bool:
    res = qa(TS_Template.format(dialo_A='\n'.join(dialo_A),
                                 dialo_B='\n'.join(dialo_B)))
    if res == "是": return True
    else: return False

def TS_Summary(summary_A: str, summary_B: str) -> bool:
    res = qa(TS_Summary_Template.format(summary_A=summary_A,
                                        summary_B=summary_B))
    if res == "是": return True
    else: return False
def Seg_summary_check4pair(Fragments: list[Fragment]) -> (list[Fragment], bool):
    for i in range(len(Fragments) - 1):
        summary_A = Fragments[i].summary
        for k in range(len(Fragments)-1, i+1, -1):
            summary_B = Fragments[k].summary
            ts_res = TS_Summary(summary_A, summary_B)
            if ts_res:
                Fragments[i].extend(Fragments[k])
                print(Fragments[i].summary)
                Fragments.pop(k)
                return Fragments, True
    return Fragments, False

def Seg_summary_check(Fragments: list[Fragment]) -> list[Fragment]:
    flag = True
    while flag:
        Fragments, flag = Seg_summary_check4pair(Fragments)
    return Fragments
def Seg_check4pair(dialo: list, Seg_res:list[list]) -> (list[list], bool):
    for i in range(len(Seg_res) - 1):
        dialo_A = [dialo[j] for j in Seg_res[i]]
        for k in range(i+1, len(Seg_res)-1):
            dialo_B = [dialo[j] for j in Seg_res[k]]
            ts_res = TS(dialo_A, dialo_B)
            if ts_res:
                Seg_res[i].extend(Seg_res[k])
                Seg_res.pop(k)
                return Seg_res, True
    return Seg_res, False
def Seg_check(dialo: list, Seg_res:list[list]) -> list[list]:
    flag = True
    while flag:
        Seg_res, flag = Seg_check4pair(dialo, Seg_res)
    for item in Seg_res: item.sort()
    return Seg_res

def DS(dialo: list) -> str:
    res = qa(DS_Template.format(dialo = dialo))
    return res

# 长对话主题分割
def long_DTS(dialo: list) -> list[list]:
    Seg_res = []
    pos_now = 0

    Seg_res = long_DTS_continue(dialo, Seg_res, pos_now)
    return Seg_res

def long_DTS_continue(dialo: list, Seg: list[list], pos_now: int) -> list[list]:
    window_size = 10
    Seg_res = Seg
    print(f"分割进度：{round(pos_now / len(dialo) * 100, 3)}%")

    while pos_now < len(dialo):

        pos_new = pos_now + min([window_size, len(dialo) - pos_now])

        dialo_now = [dialo[i] for i in range(pos_now, pos_new)]

        Seg_res_now = DTS(dialo_now)
        Seg_res_now = Seg_check(dialo_now, Seg_res_now)
        Seg_res_now = [[sub_item+pos_now for sub_item in item] for item in Seg_res_now]

        for item in Seg_res:
            dialo_A = [dialo[i] for i in item]
            del_pos = []
            for k in range(len(Seg_res_now)):
                dialo_B = [dialo[i] for i in Seg_res_now[k]]
                ts_res = TS(dialo_A, dialo_B)
                if ts_res:
                    item.extend(Seg_res_now[k])
                    dialo_A.extend(dialo_B)
                    del_pos.append(k)
            for k in range(len(del_pos)-1, -1, -1): Seg_res_now.pop(del_pos[k])

        Seg_res.extend(Seg_res_now)
        pos_now = copy.copy(pos_new)
        print(f"分割进度：{round(pos_new / len(dialo) * 100, 3)}%")

    return Seg_res

def Summary(dialo: list) -> str:
    window_size = 15
    pos_now = 0
    summary_now = ""

    while pos_now < len(dialo):
        pos_new = pos_now + min([window_size, len(dialo) - pos_now])
        dialo_now = [dialo[i] for i in range(pos_now, pos_new)]

        summary_now = qa(DS_Template.format(Summarized_content = summary_now,
                                            dialo = '\n'.join([item for item in dialo_now])))

        pos_now = copy.copy(pos_new)
    return summary_now



def Dialo_Summary(dialo: list) -> (list[list], list[str]):
    Seg_res = long_DTS(dialo)
    # Seg_res = Seg_check(dialo, Seg_res)

    dialo_summary = []
    for item in Seg_res:
        dialo_summary.append(qa(DS_Template.format(Summarized_content = "",
                                                   dialo = '\n'.join([dialo[i] for i in item]))))

    for i in range(len(Seg_res)):
        print(Answer_Template.format(Serial_Num = i+1,
                                     summary = dialo_summary[i],
                                     content = '\n\t'.join([dialo[j] for j in Seg_res[i]])))

    return Seg_res, dialo_summary