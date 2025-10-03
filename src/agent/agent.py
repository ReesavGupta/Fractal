from tools import get_tool_list
from memory import FractalMemory

class CodingAgent:
    llm = None
    tools = []
    memory : FractalMemory | None = None

    def __init__(self, llm ) -> None:
        if llm == "openai":
            # get users open ai key
            pass
        elif llm == "claude":
            # get users anthropic api key
            pass
        else:
            # get users gemini api key
            pass
        
        memory = FractalMemory()
        tools = get_tool_list()

    async def async_query_fractal(self, query: str = ""):
        pass
