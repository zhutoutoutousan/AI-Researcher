# openai_utils.py
import openai
import backoff
import time
import os
import tiktoken
import asyncio
from typing import Optional
import datetime
import global_state

def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

class GPTClient:
    def __init__(self, api_key: str = None, model: str = 'gpt-4o-mini-2024-07-18'):# 'gpt-4o-mini-2024-07-18'):# 'o1-mini-2024-09-12'):
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            api_url = os.getenv('API_BASE_URL')
            if api_key is None:
                raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = openai.AsyncClient(
            api_key=api_key,
            base_url=api_url,
            timeout=240.0,
            max_retries=0
        )
        self.model = model

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError),
        base=2,
        factor=3,
        max_tries=5
    )
    async def _get_response(self, messages: list, temperature: float, max_tokens: int) -> Optional[str]:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # temperature=temperature,
                # max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            raise

    # async def chat(self, 
    #               prompt: str, 
    #               system_prompt: str = None,
    #               temperature: float = 0.7,
    #               max_tokens: int = 16384) -> Optional[str]:
        
    #     messages = []
    #     if system_prompt:
    #         messages.append({
    #             "role": "system",
    #             "content": system_prompt
    #         })
        
    #     messages.append({
    #         "role": "user",
    #         "content": prompt
    #     })

    #     model_name = 'gpt-4'# if self.model.startswith('gpt-4') else self.model
    #     total_tokens = sum(count_tokens(msg["content"], model_name) for msg in messages)
    #     # print(f"Total input tokens: {total_tokens}")

    #     if total_tokens > 128000:
    #         print("Warning: Input tokens exceed model's context window")
    #         return None

    #     try:
    #         for attempt in range(3):
    #             try:
    #                 return await self._get_response(messages, temperature, max_tokens)
    #             except (openai.APIConnectionError, openai.APITimeoutError) as e:
    #                 if attempt == 2:
    #                     print(f"Final attempt failed: {str(e)}")
    #                     return None
    #                 print(f"Attempt {attempt + 1} failed, retrying after delay...")
    #                 await asyncio.sleep(5 * (attempt + 1))
    #             except Exception as e:
    #                 print(f"Unexpected error: {str(e)}")
    #                 return None
    #     except Exception as e:
    #         print(f"Error in chat method: {str(e)}")
    #         return None


    async def chat(self, 
               prompt: str, 
               system_prompt: str = None,
               temperature: float = 0.7,
               max_tokens: int = 16384,
               log_path: str = global_state.LOG_PATH) -> Optional[str]:
    
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        def log_message(header: str, timestamp: datetime.datetime, content: str):
            if not log_path:
                return
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{'*' * 25} {header} {'*' * 25}\n")
                log_file.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}]\n")
                log_file.write(f"{content}\n")

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Log the input
        log_message("Receive Task", datetime.datetime.now(), f"Receiveing the task:\n{prompt}")

        # Token check (optional)
        model_name = 'gpt-4'
        total_tokens = sum(count_tokens(msg["content"], model_name) for msg in messages)

        if total_tokens > 128000:
            print("Warning: Input tokens exceed model's context window")
            return None

        try:
            for attempt in range(3):
                try:
                    response = await self._get_response(messages, temperature, max_tokens)

                    # ✅ Log the output
                    log_message("Assistant Message", datetime.datetime.now(), f"Paper Agent: {response.strip()}")

                    return response

                except (openai.APIConnectionError, openai.APITimeoutError) as e:
                    if attempt == 2:
                        print(f"Final attempt failed: {str(e)}")
                        return None
                    print(f"Attempt {attempt + 1} failed, retrying after delay...")
                    await asyncio.sleep(5 * (attempt + 1))
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    return None
        except Exception as e:
            print(f"Error in chat method: {str(e)}")
            return None

