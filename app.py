from core.agent import Agent



agent = Agent()



if __name__=="__main__":


    session_id="user001"


    while True:


        text=input(
            "用户:"
        )


        # print(
        #     "AI:",
        #     end=""
        # )


        for token in agent.stream_chat(
            session_id,
            text
        ):

            print(
                token,
                end="",
                flush=True
            )


        print()