#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from zhipuai import ZhipuAI
from dashscope import Generation
from abc import ABC
from openai import OpenAI
import openai
from ollama import Client
from rag.nlp import is_english
from anthropic import AnthropicBedrock

class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def chat(self, system, history, gen_conf):
        raise NotImplementedError("Please implement encode method!")


class GptTurbo(Base):
    def __init__(self, key, model_name="gpt-3.5-turbo", base_url="https://api.openai.com/v1"):
        if not base_url: base_url="https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf)
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0


class MoonshotChat(GptTurbo):
    def __init__(self, key, model_name="moonshot-v1-8k", base_url="https://api.moonshot.cn/v1"):
        if not base_url: base_url="https://api.moonshot.cn/v1"
        self.client = OpenAI(
            api_key=key, base_url=base_url)
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf)
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0


class QWenChat(Base):
    def __init__(self, key, model_name=Generation.Models.qwen_turbo, **kwargs):
        import dashscope
        dashscope.api_key = key
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        from http import HTTPStatus
        if system:
            history.insert(0, {"role": "system", "content": system})
        response = Generation.call(
            self.model_name,
            messages=history,
            result_format='message',
            **gen_conf
        )
        ans = ""
        tk_count = 0
        if response.status_code == HTTPStatus.OK:
            ans += response.output.choices[0]['message']['content']
            tk_count += response.usage.total_tokens
            if response.output.choices[0].get("finish_reason", "") == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, tk_count

        return "**ERROR**: " + response.message, tk_count


class ZhipuChat(Base):
    def __init__(self, key, model_name="glm-3-turbo", **kwargs):
        self.client = ZhipuAI(api_key=key)
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            if "presence_penalty" in gen_conf: del gen_conf["presence_penalty"]
            if "frequency_penalty" in gen_conf: del gen_conf["frequency_penalty"]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf
            )
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except Exception as e:
            return "**ERROR**: " + str(e), 0


class OllamaChat(Base):
    def __init__(self, key, model_name, **kwargs):
        self.client = Client(host=kwargs["base_url"])
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            options = {"temperature": gen_conf.get("temperature", 0.1),
                       "num_predict": gen_conf.get("max_tokens", 128),
                       "top_k": gen_conf.get("top_p", 0.3),
                       "presence_penalty": gen_conf.get("presence_penalty", 0.4),
                       "frequency_penalty": gen_conf.get("frequency_penalty", 0.7),
                       }
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                options=options
            )
            ans = response["message"]["content"].strip()
            return ans, response["eval_count"] + response.get("prompt_eval_count", 0)
        except Exception as e:
            return "**ERROR**: " + str(e), 0


class XinferenceChat(Base):
    def __init__(self, key=None, model_name="", base_url=""):
        self.client = OpenAI(api_key="xxx", base_url=base_url)
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf)
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0

class AwsClaude3(Base):
    def __init__(self, key, model_name='claude-3-opus-20240229', **kwargs):
        aws_access = key.split(":")
        if not key or len(aws_access) < 2:
            return
        if len(aws_access) == 2:
            aws_access.append("us-west-2")
        self.client = AnthropicBedrock(
            aws_access_key=aws_access[0],
            aws_secret_key=aws_access[1],
            aws_region=aws_access[2],
        )
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        try:
            if "presence_penalty" in gen_conf: del gen_conf["presence_penalty"]
            if "frequency_penalty" in gen_conf: del gen_conf["frequency_penalty"]
            if system: gen_conf["system"] = system
            if not gen_conf["max_tokens"]: gen_conf["max_tokens"] = 128
            response = self.client.messages.create(
                model= self.model_name,
                messages=history,
                **gen_conf
            )
            ans = response.content[0].text.strip()
            if response.stop_reason.strip() == "max_tokens":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.input_tokens + response.usage.output_tokens
        except Exception as e:
            return "**ERROR**: " + str(e), 0
