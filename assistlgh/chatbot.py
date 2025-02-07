# coding=utf-8
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import requests

OLLAMA_HOST = 'http://100.95.82.15:11434'
GENERATE_ENDPOINT = OLLAMA_HOST+'/api/generate'
CHAT_ENDPOINT = OLLAMA_HOST+'/api/chat'

def generate_response(prompt, model= 'deepseek-r1:8b', stream= False):
    """
    生成文本响应
    :param prompt: 输入的提示文本
    :param model: 使用的模型名称（默认llama2）
    :param stream: 是否使用流式响应（默认False）
    :return: 生成的响应文本
    """
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': stream
    }

    try:
        response = requests.post(GENERATE_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json().get('response', '')
    except requests.exceptions.RequestException as e:
        return "API请求错误:"+ str(e)

def chat_completion(messages, model= 'deepseek-r1:8b', stream= False):
    """
    聊天对话接口
    :param messages: 消息历史列表（格式：[{'role': 'user', 'content': '你好'}]
    :param model: 使用的模型名称（默认deepseek-r1:8b）
    :param stream: 是否使用流式响应（默认False）
    :return: 生成的响应内容
    """
    payload = {
        'model': model,
        'messages': messages,
        'stream': stream
    }

    try:
        response = requests.post(CHAT_ENDPOINT, json=payload)
        response.raise_for_status()

        if stream:
            full_response = []
            for line in response.iter_lines():
                if line:
                    chunk = response.json()
                    full_response.append(chunk.get('message', {}).get('content', ''))
            return ''.join(full_response)
        else:
            return response.json().get('message', {}).get('content', '')

    except requests.exceptions.RequestException as e:
        return "API请求错误: "+ str(e)
    except Exception as e:
        return "处理响应时发生错误:"+ str(e)
import sys
import pandas as pd
import time
from glob import glob
import os
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) < 2:
        raise("请至少输入1个参数")
    else:
        mode = sys.argv[1]
    if mode == 'quick':
        if len(sys.argv)==4:
            prompt=sys.argv[2]
            model=sys.argv[3]
            response = generate_response(prompt,model)
        elif len(sys.argv)==3:
            prompt=sys.argv[2]
            response = generate_response(prompt)
        else:
            raise("参数过多")
        print(response)
    else:
        previous_chat = []
        if mode == 'new':
            historytime = time.time()
            print("starting new chat...")
        elif mode == 'continue':
            historyfile = glob(parent_dir+'/*.log')[-1]
            historytime = float(historyfile.split('chat_history')[-1].split('.log')[0])    
            print("loading chat history "+str(historytime)+'...')
            chat_history = pd.read_csv(parent_dir+'/chat_history'+str(historytime)+'.log',sep='\t')
            for items in chat_history.iloc:
                print(items["role"],':',items["content"])
                previous_chat.append({"role": items["role"], "content": items["content"]})
        while True:
            prompt = input("user:")
            if prompt == "/bye":
                print("再见！")
                break
            previous_chat.append({"role": "user", "content": prompt})
            chat_response = chat_completion(previous_chat)
            thinking = chat_response.split('\n')[1]
            msg = chat_response.split('\n')[-1]
            previous_chat.append({"role": "assistant", "content": msg,"think":thinking})
            print('assistant is thinking...\n')
            print(thinking)
            print('assistant:',msg)
            pd.DataFrame(previous_chat).to_csv(parent_dir+'/chat_history'+str(historytime)+'.log',sep='\t',index=False)
