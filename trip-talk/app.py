import gradio as gr
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from chains.guide_chain import generate_guide
from graph import build_graph

# 환경 변수 로드
load_dotenv()

# 전역 그래프 인스턴스 (상태 비저장 로직, 상태는 세션별로 전달됨)
app_graph = build_graph()

async def generate_context(loc, sit):
    """가이드를 생성하고 세션 상태 컨텍스트를 초기화합니다."""
    if not loc or not sit:
        err = {"error": "Please enter location and situation."}
        return err, {}, {}, {}, {}
    
    print(f"Generating guide for {loc} - {sit}...")
    try:
        # 비동기 함수 호출
        context_data = await generate_guide(loc, sit)
    except Exception as e:
        print(f"Error in generate_guide: {e}")
        err = {"error": f"Error generating guide: {str(e)}"}
        return err, {}, {}, {}, {}
    
    guide_data = context_data.get("guide", {})
    
    
    # 각 항목 분리 및 마크다운 변환
    flow_list = guide_data.get("conversation_flow", [])
    flow_md = "\n\n".join(flow_list) if flow_list else "대화 흐름이 없습니다."
    
    speaking_list = guide_data.get("speaking_expressions", [])
    speaking_md = "\n".join([f"- {item}" for item in speaking_list]) if speaking_list else "표현이 없습니다."
    
    listening_list = guide_data.get("listening_expressions", [])
    listening_md = "\n".join([f"- {item}" for item in listening_list]) if listening_list else "표현이 없습니다."
    
    vocab_list = guide_data.get("focused_vocabulary", [])
    vocab_md = "\n".join([f"- {item}" for item in vocab_list]) if vocab_list else "단어가 없습니다."
    
    return flow_md, speaking_md, listening_md, vocab_md, context_data

async def chat_response(message, history, context, loc, sit):
    """
    LangGraph를 사용하여 사용자 채팅 메시지를 처리합니다.
    history: Gradio의 [{"role": "user", "content": ...}, ...] 리스트
    """
    print(f"Gradio Version: {gr.__version__}")
    
    # history가 None일 경우 빈 리스트로 초기화
    if history is None:
        history = []
        
    if not context:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "먼저 가이드를 생성해주세요! (버튼 클릭)"})
        yield history, ""
        return
    
    # Gradio 기록을 LangChain 메시지로 변환
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        else:
            # Fallback for tuple format if mixed
            if len(msg) >= 2:
                messages.append(HumanMessage(content=str(msg[0])))
                if msg[1]:
                    messages.append(AIMessage(content=str(msg[1])))
    
    messages.append(HumanMessage(content=message))
    
    # 그래프 실행
    inputs = {
        "messages": messages,
        "context_data": context,
        "location": loc,
        "situation": sit
    }
    
    # 사용자 메시지 먼저 표시
    history.append({"role": "user", "content": message})
    yield history, ""
    
    try:
        # 비동기 그래프 호출
        result = await app_graph.ainvoke(inputs)
        full_response = result["messages"][-1].content
        
        # 스트리밍 시뮬레이션 (한 글자씩 출력)
        history.append({"role": "assistant", "content": ""})
        for i in range(len(full_response)):
            history[-1]["content"] = full_response[:i+1]
            yield history, ""
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})
        yield history, ""

# Google Places 도구 초기화
from tools.google_places import GooglePlacesTool
place_tool = GooglePlacesTool()

def update_suggestions(query):
    """검색어 변경 시 장소 추천 목록 업데이트"""
    if not query or len(query) < 2:
        return gr.update(choices=[], visible=False)
    
    try:
        results = place_tool.search_places(query)
        # Dropdown choices: ["Main Text (Full Text)", ...]
        choices = [f"{item['main_text']} ({item['description']})" for item in results]
        return gr.update(choices=choices, visible=True)
    except Exception as e:
        print(f"Suggestion Error: {e}")
        return gr.update(choices=[], visible=False)

def select_place(selected_text):
    """추천 장소 선택 시 장소 입력창 채우기"""
    if not selected_text:
        return gr.update()
    
    # "Main Text (Full Text)" 형식에서 Description 부분 추출 또는 전체 사용
    # 여기서는 괄호 포함 전체 텍스트를 사용하거나, 파싱해서 정제할 수 있음.
    # 사용 편의를 위해 전체 텍스트 사용
    return selected_text

# UI 레이아웃
with gr.Blocks(title="TripTalker", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✈️ TripTalker: 실전 여행 회화 시뮬레이터")
    
    # 세션 상태
    context_state = gr.State({})
    
    with gr.Row():
        with gr.Column(scale=4 ,min_width=400):
            # Google Places Autocomplete
            gr.Markdown("### 📍 장소 검색")
            search_input = gr.Textbox(label="장소 검색", placeholder="예: 도쿄 디즈니, 오사카 라면...", show_label=False)
            suggestion_dropdown = gr.Dropdown(label="추천 장소", visible=False, interactive=True)
            
            gr.Markdown("---")
            location_input = gr.Textbox(label="장소 / 국가 (자동 입력됨)", placeholder="직접 입력하거나 위에서 검색하세요")
            situation_input = gr.Textbox(label="상황", placeholder="예: 고수 빼고 매운 라면 주문하기")
            btn_start = gr.Button("1. 가이드 받기 & 시작", variant="primary")
            
            # 이벤트 연결 (UI 내부 정의)
            search_input.change(
                fn=update_suggestions,
                inputs=search_input,
                outputs=suggestion_dropdown
            )
            
            suggestion_dropdown.change( # select 대신 change 사용 (Dropdown 값 변경 시)
                fn=select_place,
                inputs=suggestion_dropdown,
                outputs=location_input
            )

            # 1. 대화 흐름 (가장 중요)
            with gr.Tabs():
                with gr.TabItem("📖 대화 흐름"):
                    # JSON 대신 Markdown 사용
                    flow_output = gr.Markdown("가이드 버튼을 누르면 내용이 표시됩니다.")
            
            # 2. 표현 (Speaking / Listening)
                with gr.TabItem("🗣️ 주요 표현"):
                    with gr.Accordion("말하기 (Speaking)", open=True):
                        speaking_output = gr.Markdown("- 표현이 여기에 표시됩니다.")
                    with gr.Accordion("듣기 (Listening)", open=True):
                        listening_output = gr.Markdown("- 표현이 여기에 표시됩니다.")
            
            # 3. 단어 및 메뉴 (통합)
                with gr.TabItem("word 단어장"):
                    vocab_output = gr.Markdown("추천 단어가 표시됩니다.")
            
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(label="시뮬레이션", height=500)
            msg_input = gr.Textbox(label="메시지 입력", placeholder="여기에 입력하세요... (엔터로 전송)")
            clear = gr.Button("대화 지우기")

    # 이벤트 연결
    btn_start.click(
        generate_context,
        inputs=[location_input, situation_input],
        outputs=[flow_output, speaking_output, listening_output, vocab_output, context_state]
    )
    
    msg_input.submit(
        chat_response,
        inputs=[msg_input, chatbot, context_state, location_input, situation_input],
        outputs=[chatbot, msg_input]
    )

if __name__ == "__main__":
    demo.launch()
