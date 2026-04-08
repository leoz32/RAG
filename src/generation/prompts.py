def build_non_rag_prompt(question: str) -> str:
    return (
        "你是一个高等教育教学助手，请直接回答用户问题，保持学术表达清晰、简洁、规范。\n\n"
        f"问题：{question}\n\n"
        "回答："
    )


def build_rag_prompt(question: str, contexts: str) -> str:
    return (
        "你是一个高等教育教学助手。\n"
        "请严格依据给定教学资料回答问题。\n"
        "如果资料中没有明确答案，请明确回答“未知”。\n"
        "不要编造，不要补充资料外信息，不要使用你自身常识替代资料内容。\n\n"
        f"教学资料：\n{contexts}\n\n"
        f"问题：{question}\n\n"
        "回答："
    )
