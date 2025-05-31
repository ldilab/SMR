import logging
import ollama
import re
import time
import tiktoken

from openai import OpenAI
from transformers import AutoTokenizer

class LLM:
    '''
    A high-level interface for language models.

    Depending on the model name, it automatically selects 
    either an OpenAI-based model (OpenAIModel) or 
    a HuggingFace-based model (HFModel).
    '''
    OPENAI_MODEL_LIST = ['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini', 'o3-mini', 'o3-mini-high']
    OLLAMA_MODEL_LIST = [m.model for m in ollama.list().models]

    def __init__(self, model_name: str, tokenizer=None, openai_token: str = None, hf_token: str = None):
        '''
        Determine which model type to instantiate 
        based on the given model_name.

        Args:
            model_name (str): Name of the model to be used.
        '''
        if model_name in self.OPENAI_MODEL_LIST:
            self.llm = OpenAIModel(model_name, openai_token)
        elif model_name in self.OLLAMA_MODEL_LIST:
            self.llm = OllamaModel(model_name, tokenizer)
        else:
            self.llm = HFModel(model_name, hf_token)

    def generate(self, user_prompt: str, system_prompt: str = '', **kwargs) -> str:
        '''
        Generate text from the model.

        Args:
            user_prompt (str): The user prompt to provide to the model.
            system_prompt (str, optional): Additional context or system 
                                           prompt for conversation.
            **kwargs: Additional arguments to forward to the underlying 
                      model's generation method.

        Returns:
            str: The generated text.
        '''
        return self.llm.generate(user_prompt, system_prompt, **kwargs)


class OpenAIModel:
    '''
    OpenAI-based model class to generate text from OpenAI's Chat APIs.
    Includes retry logic to handle transient errors and returns both text and token count.
    '''

    def __init__(self,
                 model_name: str,
                 openai_token: str,
                 max_retry: int = 5,
                 min_retry_wait: float = 0.1,
                 max_retry_wait: float = 5.0):
        '''
        Initialize an OpenAI model client with basic retry configuration.

        Args:
            model_name (str): Name of the OpenAI model (e.g., 'gpt-4').
            openai_token (str): Your OpenAI API key.
            max_retry (int, optional): Maximum number of retries upon failure.
            min_retry_wait (float, optional): Minimum wait before retrying.
            max_retry_wait (float, optional): Maximum wait before retrying.
        '''
        self.model_name = model_name
        self.client = OpenAI(api_key=openai_token)
        self.max_retry = max_retry
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait

    def generate(self,
                 user_prompt: str,
                 system_prompt: str = '',
                 remove_space: bool = True,
                 **kwargs) -> tuple[str, int]:
        '''
        Generate a response and token count from an OpenAI language model.

        Returns:
            tuple:
                - str: Generated text.
                - int: Number of tokens in the generated text.
        '''
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        retry_cnt = 0
        retry_wait = self.min_retry_wait

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(e, flush=True)
                if retry_cnt >= self.max_retry:
                    logging.error(
                        f'\n[System Prompt] {system_prompt}'
                        f'\n[User Prompt] {user_prompt}'
                        f'\n[Error] {e}'
                    )
                    return '', 0
                time.sleep(retry_wait)
                retry_cnt += 1
                retry_wait = min(retry_wait * 2, self.max_retry_wait)
                continue
            break

        result = response.choices[0].message.content
        if remove_space:
            result = ' '.join(result.split())

        # Count generated tokens using tiktoken
        encoding = tiktoken.encoding_for_model(self.model_name)
        token_count = len(encoding.encode(result))

        return result, token_count

class OllamaModel:
    '''
    Ollama-based model class to generate text from Ollama's Chat APIs,
    and count the number of generated tokens.
    '''

    def __init__(self,
                 model_name: str,
                 tokenizer: str = None,
                 max_retry: int = 5,
                 min_retry_wait: float = 0.1,
                 max_retry_wait: float = 5.0):
        '''
        Initialize an Ollama model client.

        Args:
            model_name (str): Name of the Ollama model (e.g., 'llama3.1:8b').
        '''
        self.model_name = model_name
        self.max_retry = max_retry
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

    def generate(self,
                 user_prompt: str,
                 system_prompt: str = '',
                 remove_space: bool = True,
                 temperature: float = 0.0,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 seed: int = 42,
                 repeat_penalty: float = 1.2,
                 **kwargs) -> tuple[str, int]:
        '''
        Generate a response from an Ollama language model and count tokens.

        Returns:
            tuple:
              - str: Generated text (with <think> blocks and extra whitespace removed)
              - int: Number of tokens in the generated text
        '''
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})
        messages.append({'role': 'assistant', 'content': '<think>Okay, I think I have finished thinking.</think> {\n'})
        #'<|beginning of thinking|>\nOkay, I think I have finished thinking.\n<|end of thinking|> {\n'

        # Count tokens in the messages in prompt
        token_count_input = 0
        for message in messages:
            token_count_input += len(self.tokenizer.encode(message['content'], add_special_tokens=False))
        num_ctx = min(token_count_input + 3500, 100000)
        retry_cnt = 0
        retry_wait = self.min_retry_wait
        while True:
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    options={'num_ctx': num_ctx,
                             'num_predict': 3000,
                             'temperature': temperature,
                             'top_k': top_k,
                             'top_p': top_p,
                             'seed': seed,
                             'repeat_penalty': repeat_penalty},
                    **kwargs,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error(f"Ollama chat error: {e}")
                if retry_cnt >= self.max_retry:
                    return '', 0
                time.sleep(retry_wait)
                retry_cnt += 1
                retry_wait = min(retry_wait * 2, self.max_retry_wait)
                continue
            break

        # Extract the text content from the model's response
        text = response['message']['content']
        text = '{\n' + text
        # Count generated tokens
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_count = len(token_ids)
        # remove <think> blocks and extra whitespace
        if '<think>' in text:
            print(f'Found <think> in the response: {text}')
        response_wo_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        if '</think>' in response_wo_think:
            response_wo_think = response_wo_think.split('</think>')[0]
        if remove_space:
            response_wo_think = ' '.join(response_wo_think.split())

        return response_wo_think, token_count, text


class HFModel:
    '''
    HuggingFace-based model class. 
    (Implementation details would vary depending on the specific HF model usage.)
    '''

    def __init__(self, model_name: str, hf_token : str):
        '''
        Initialize a HuggingFace model (not implemented here).

        Args:
            model_name (str): Name of the HuggingFace model.
        '''
        pass

    def generate(self, 
                 user_prompt: str, 
                 system_prompt: str = '', 
                 **kwargs) -> str:
        '''
        Generate a response from a HuggingFace-based model (not implemented).

        Args:
            user_prompt (str): The user prompt to provide to the model.
            system_prompt (str, optional): System-level prompt or context.
            **kwargs: Additional keyword arguments for text generation.

        Returns:
            str: Placeholder response indicating unimplemented functionality.
        '''
        pass
