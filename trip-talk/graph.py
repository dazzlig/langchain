from langgraph.graph import StateGraph, END
from state import TripTalkerState
from agents.router import router_node
from agents.personas import clerk_node, tutor_node

def build_graph():
    workflow = StateGraph(TripTalkerState)
    
    # 노드 추가
    workflow.add_node("router", router_node)
    workflow.add_node("clerk", clerk_node)
    workflow.add_node("tutor", tutor_node)
    
    # 진입점 설정
    workflow.set_entry_point("router")
    
    # 라우터에서의 조건부 엣지 정의
    workflow.add_conditional_edges(
        "router",
        lambda x: x["user_intent"],
        {
            "clerk": "clerk",
            "tutor": "tutor"
        }
    )
    
    # END로 가는 엣지 또는 다시 Router로?
    # 보통 채팅에서는 사용자에게 반환하고, 그 다음 사용자가 다시 입력합니다.
    # LangGraph에서는 한 번의 턴 후에 실행을 종료합니다.
    workflow.add_edge("clerk", END)
    workflow.add_edge("tutor", END)
    
    return workflow.compile()
