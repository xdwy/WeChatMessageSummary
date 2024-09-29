import json
from io import StringIO

import chardet
import pandas as pd
import uvicorn
from fastapi import FastAPI, Body, UploadFile
from fastapi.responses import StreamingResponse


from customResponse import CustomResponse
from service import process_wechat

app = FastAPI()


@app.post("/v1/wechat/process/message", summary="处理微信信息")
async def process_message(file: UploadFile):
    try:
        # 读取文件的字节内容
        content = await file.read()
        # 自动检测文件编码
        result = chardet.detect(content)
        encoding = result['encoding']
        # 使用检测到的编码解码
        csv_data = StringIO(content.decode(encoding=encoding, errors='ignore'))  # 忽略非法字符
        data = pd.read_csv(csv_data)
        res = process_wechat(data)
        return CustomResponse(code=200, msg="数据处理成功", data=res)
    except Exception as e:
        return CustomResponse(code=500, msg=str(e), data=None)


if __name__ == "__main__":
    uvicorn.run(port=9700, app=app, host="0.0.0.0")