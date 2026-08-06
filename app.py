from core.graph import graph


if __name__ == "__main__":

    result = graph.invoke(
        {
            "user_input": "你好！"
        }
    )

    print("\n========== 最终结果 ==========")

    print(result["response"])