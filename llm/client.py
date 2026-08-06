from openai import OpenAI

from config import Config



class LLMClient:


    def __init__(self):

        self.client = OpenAI(

            api_key=Config.OPENAI_API_KEY,

            base_url=Config.OPENAI_BASE_URL
        )



    def stream_chat(
            self,
            prompt:str
    ):


        response = self.client.chat.completions.create(

            model=Config.LLM_MODEL,

            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ],

            stream=True
        )

        for chunk in response:
            # 直接检查choices[0].delta.content是否存在
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content