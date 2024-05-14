from toolkit.langchain import llms, memories, chains, prompts
from langchain.chains import ConversationChain


def create_conversation_chain(
    prompt=prompts.general_prompt,
    llm=llms.llm_openai,
    memory=memories.ConversationBufferMemory(ai_prefix="AI Assistant"),
    verbose=False,
):
    return chains.ConversationChain(
        prompt=prompt,
        llm=llm,
        memory=memory,
        verbose=verbose,
    )


def get_conversation_chain_response(human_msg, conversation_chain: ConversationChain):
    result = conversation_chain.invoke(human_msg)['response']
    return result


my_conversation_chain = create_conversation_chain()

# human_message = "Hi there"
# result = get_conversation_chain_response(human_message, my_conversation_chain)
# print(result)
