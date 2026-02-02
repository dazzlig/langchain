from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class TripTalkerState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context_data: Dict[str, Any]  # 병합된 가이드/메뉴 데이터 저장
    current_persona: str          # 'clerk' (점원) 또는 'tutor' (튜터)
    user_intent: str              # 'role_play' (연기) 또는 'question' (질문)
    location: str
    situation: str
