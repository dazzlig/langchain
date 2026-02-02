from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

from state import TripTalkerState

class RouteQuery(BaseModel):
    """사용자 질문을 가장 관련성 높은 노드로 라우팅합니다."""
    target: Literal["clerk", "tutor"] = Field(
        ...,
        description="라우팅할 대상 페르소나입니다. 역할극/연기는 'clerk', 언어/상황에 대한 질문은 'tutor'입니다."
    )

def router_node(state: TripTalkerState):
    """
    사용자가 역할극(Role-play)을 하고 있는지 질문(Question)을 하고 있는지 결정합니다.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 구조화된 출력을 사용하는 LLM을 이용한 단순 라우터
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)
    
    system = """You are a router agent for a language learning app.
    Your job is to determine if the user's message is:
    1. A 'role-play' line (e.g., "I would like a coffee", "How much is this?"). They are talking TO the character in the scenario. -> Route to 'clerk'
    2. A 'question' about the language or situation (e.g., "How do I say 'receipt' in Korean?", "Is this polite?"). They are talking TO the tutor. -> Route to 'tutor'
    
    Context:
    Location: {location}
    Situation: {situation}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}")
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "question": last_message.content, 
        "location": state.get("location", "General"),
        "situation": state.get("situation", "General")
    })
    
    return {"user_intent": result.target, "current_persona": result.target}
