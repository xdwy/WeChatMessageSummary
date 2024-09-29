import json


class CustomResponse:
    def __init__(self, code, msg, data=None):
        """
        初始化自定义响应类
        :param code: int, HTTP状态码
        :param msg: str, 响应消息
        :param data: any, 可选的响应数据，默认是None
        """
        self.code = code
        self.msg = msg
        self.data = data

    def to_dict(self):
        """
        将响应对象转换为字典
        :return: dict, 包含响应数据的字典
        """
        response = {
            'code': self.code,
            'msg': self.msg,
            'data': self.data
        }
        return response

    def __str__(self):
        """
        自定义字符串表示
        :return: str, 响应对象的字符串表示
        """
        return json.dumps(self.to_dict())