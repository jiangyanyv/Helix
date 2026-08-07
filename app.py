from time import sleep

from modelscope.server.models import output

from core.agent import Agent
from voice.tts.tts_worker import TTSWorker
from utils.logger import logger

agent = Agent()



if __name__=="__main__":


    session_id="user001"

    tts = TTSWorker()
    tts.start()

    while True:


        text=input(
            "输入:"
        )


        for token in agent.stream_chat(
            session_id,
            text
        ):
            pass

        sleep(1)
        print()