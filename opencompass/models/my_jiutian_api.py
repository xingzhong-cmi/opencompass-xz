import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList
from opencompass.registry import MODELS
from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class JiutianAPI(BaseAPIModel):
    """Model wrapper around Jiutian API.

    Args:
        path (str): The name of Jiutian API model.
            e.g. `jiutian-v3`
        key (str): Authorization key for Jiutian API.
        url (str): The API endpoint URL for Jiutian.
        query_per_second (int): The maximum queries allowed per second.
            Defaults to 2.
        max_seq_len (int): Maximum sequence length supported by the model.
            Defaults to 2048.
        meta_template (Dict, optional): The model's meta prompt template.
        retry (int): Number of retries if API call fails. Defaults to 2.
        system_prompt (str): System prompt to prepend to inputs. Defaults to ''.
    """

    def __init__(
        self,
        path: str,
        key: str,
        url: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        system_prompt: str = '',
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry
        )
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {key}'  # 九天API认证头
        }
        self.url = url  # 九天API端点地址
        self.model = path  # 九天模型名称（如jiutian-v3）
        self.system_prompt = system_prompt  # 系统提示词

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
            max_out_len (int): Maximum length of output.

        Returns:
            List[str]: Generated results.
        """
        # 并发处理输入列表
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs, [max_out_len] * len(inputs))
            )
        self.flush()  # 刷新速率控制
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        """Generate result for a single input."""
        assert isinstance(input, (str, PromptList)), f"Invalid input type: {type(input)}"

        # 转换输入格式为API要求的messages结构
        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            msg_buffer, last_role = [], None
            for item in input:
                # 转换角色：OpenCompass的"BOT"对应API的"assistant"
                item_role = 'assistant' if item['role'] == 'BOT' else 'user'
                if item_role != last_role and last_role is not None:
                    # 拼接同角色的消息
                    messages.append({
                        'content': '\n'.join(msg_buffer),
                        'role': last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item['prompt'])
                last_role = item_role
            # 添加最后一组消息
            if msg_buffer:
                messages.append({
                    'content': '\n'.join(msg_buffer),
                    'role': last_role
                })

        # 插入系统提示词（若有）
        if self.system_prompt:
            messages.insert(0, {'role': 'system', 'content': self.system_prompt})

        # 构造API请求参数
        data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_out_len  # 控制输出长度
        }

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()  # 速率控制：获取请求许可
            try:
                # 发送API请求
                raw_response = requests.post(
                    url=self.url,
                    headers=self.headers,
                    json=data,
                    timeout=60
                )
            except Exception as err:
                self.logger.warning(f"Request failed: {str(err)}")
                time.sleep(2)
                continue
            finally:
                self.release()  # 释放请求许可

            # 解析响应
            try:
                response = raw_response.json()
            except Exception as err:
                self.logger.error(f"Failed to parse response: {str(err)}")
                response = None

            # 处理连接错误
            if response is None:
                self.logger.warning("Connection error, retrying...")
                self.wait()  # 等待后重试
                continue

            # 处理成功响应
            if raw_response.status_code == 200:
                try:
                    generated_text = response['choices'][0]['message']['content'].strip()
                    self.logger.debug(f"Generated: {generated_text}")
                    return generated_text
                except KeyError as err:
                    self.logger.error(f"Invalid response structure: {str(err)}")
                    max_num_retries += 1
                    continue

            # 处理认证错误
            elif raw_response.status_code == 401:
                self.logger.error("API key is invalid or expired")
                max_num_retries += 1
                continue

            # 处理请求错误（如参数错误）
            elif raw_response.status_code == 400:
                self.logger.error(f"Bad request: {response.get('error', 'Unknown error')}")
                return "Request failed due to invalid parameters"

            # 处理限流错误
            elif raw_response.status_code == 429:
                self.logger.warning("Rate limited, waiting...")
                time.sleep(5)  # 限流时等待更长时间
                continue

            # 其他错误
            else:
                self.logger.error(
                    f"API request failed with status {raw_response.status_code}: {response}"
                )
                max_num_retries += 1
                time.sleep(1)

        # 重试耗尽仍失败
        raise RuntimeError(f"Failed after {self.retry} retries. Last response: {raw_response.text}")
