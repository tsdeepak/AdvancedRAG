from dotenv import load_dotenv
from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool

load_dotenv()
llm = OpenAI()


def write_haiku(topic: str) -> str:
    """
    Writes a haiku about a given topic
    """
    return llm.complete(f"wrtie me a haiku about {topic}")


def count_characters(text: str) -> int:
    """
    Counts the number of characters in a text
    """
    return len(text)


if __name__ == "__main__":
    print("**** Hello Agents LlamaIndex ****")

    tool1 = FunctionTool.from_defaults(fn=write_haiku, name="WriteHaiku")
    tool2 = FunctionTool.from_defaults(fn=count_characters, name="CountChars")

    agent = ReActAgent.from_tools(tools=[tool1, tool2], llm=llm, verbose=True)
    res = agent.query(
        "Write me a haiku about cricket and then count the characters in it"
    )

    print(res)
