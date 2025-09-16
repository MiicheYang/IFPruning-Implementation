import openai
from openai import AzureOpenAI, OpenAI
import time
from typing import List

from tenacity import (
    retry,
    wait_chain,
    wait_fixed
)


# @retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] +
#                        [wait_fixed(2) for i in range(2)] +
#                        [wait_fixed(3)]))
def completion_with_backoff_mcopenai(**kwargs):
    client = AzureOpenAI(
        api_version="2023-12-01-preview",
        azure_endpoint="https://0212openai.openai.azure.com/",
        api_key="352d7f1511084d6d8a37f7214c5eb528",
    )
    result = client.chat.completions.create(
        model="gpt4-azure-0212",
        **kwargs,
    )
    return result

def completion_with_backoff_mcopenai_chatgpt(**kwargs):
    while True:
        try:
            client = AzureOpenAI(
                api_version="2024-02-15-preview",
                azure_endpoint="https://chatgpt-0125.openai.azure.com/",
                api_key="dca4cf09329941098c51a8ca09c036ef",
            )
            result = client.chat.completions.create(
                model="chatgpt_0125",
                **kwargs,
            )
            break
        except Exception as e:
            reason = e.body['code']
            if reason == 'content_filter':
                return None
            time.sleep(3)
    return result


def openai_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content": prompt}]
    
    response = completion_with_backoff_mcopenai(
        # engine=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        **kwargs,
    )
    contents = []
    for choice in response.choices:
        # Check if the response is valid
        # if choice['finish_reason'] not in ['stop', 'length']:
        if choice.finish_reason not in ['stop', 'length']:
            raise ValueError(f"OpenAI Finish Reason Error: {choice['finish_reason']}")
        contents.append(choice.message.content)

    return contents
     