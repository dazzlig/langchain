import os
import json
from typing import Annotated, List, TypedDict, Dict, Any, Literal
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# --- Helper Functions ---
def save_step_to_file(run_id, step_name, result):
    directory = f"runs/{run_id}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Simple file naming strategy for this example
    import time
    timestamp = int(time.time())
    filename = f"{directory}/{timestamp}_{step_name}.json"
    
    data = {
        "step_name": step_name,
        "result": result
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- LLM & Tools ---
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
# Ensure Tavily API Key is set in env or handle error
try:
    search_tool = TavilySearchResults(k=3)
except Exception:
    print("Warning: Tavily API Key missing, search will fail if called.")
    search_tool = None # Handle appropriately in node

# ==========================================
# 1. Research Subgraph
# ==========================================
class ResearchState(TypedDict):
    topic: str
    logs: Annotated[List[BaseMessage], add_messages]
    raw_data: str
    quality: str
    retry_count: int
    run_id: str # Added to pass run_id down

def research_execute_node(state: ResearchState):
    print(f"[Research] 정보 수집 중... Topic: {state['topic']}")
    topic = state["topic"]
    
    try:
        if search_tool:
            results = search_tool.invoke(topic)
            content = "\\n".join([r["content"] for r in results])
        else:
            content = "검색 도구를 사용할 수 없습니다 (API Key Missing)."
    except Exception as e:
        content = f"검색 실패: {str(e)}"
        
    return {
        "raw_data": content, 
        "logs": [AIMessage(content=f"검색 완료: {len(content)}자", name="researcher")]
    }

def research_reflect_node(state: ResearchState):
    print("[Research Sub] 정보 충분성 평가 중...")
    
    chain = ChatPromptTemplate.from_template(
        """당신은 엄격한 연구 팀장입니다. 수집된 자료가 주제 '{topic}'을 설명하기에 충분한지 평가하세요.
        
        [수집된 자료]
        {data}
        
        자료가 주제를 포괄적으로 설명하면 'PASS', 부족하거나 편향되었다면 'FAIL'이라고만 답하세요.
        """
    ) | llm | StrOutputParser()
    
    evaluation = chain.invoke({"topic": state["topic"], "data": state["raw_data"]})
    quality = "PASS" if "PASS" in evaluation else "FAIL"
    
    print(f"      ㄴ 평가 결과: {quality}")
    return {"quality": quality, "logs": [AIMessage(content=f"평가 결과: {quality}", name="evaluator")]}

def research_revise_node(state: ResearchState):
    print(" [Research] 추가 검색(보완) 수행 중...")
    topic = state["topic"]
    current_data = state["raw_data"]
    
    query_chain = ChatPromptTemplate.from_template(
        """당신은 노련한 리서처입니다.
        주제 '{topic}'에 대해 현재 수집된 자료가 충분하지 않습니다.
        
        [현재 자료]
        {data}
        
        위 자료에서 빠진 내용이나 더 구체적인 정보가 필요한 부분을 파악하여,
        검색 엔진에 입력할 '구체적인 추가 검색어' 1개를 제안해주세요. (설명 없이 검색어만 출력)
        """
    ) | llm | StrOutputParser()
    
    new_query = query_chain.invoke({"topic": topic, "data": current_data[:2000]})
    print(f"      ㄴ생성된 추가 검색어: '{new_query}'")
    
    try:
        if search_tool:
            search_results = search_tool.invoke(new_query)
            new_content = "\\n".join([f"- {r['content']}" for r in search_results])
        else:
            new_content = "검색 도구 없음"
    except Exception as e:
        new_content = f"추가 검색 실패: {str(e)}"
        
    combined_data = current_data + f"\\n\\n[추가 검색 결과 ({new_query})]:\\n" + new_content
    
    return {
        "raw_data": combined_data, 
        "retry_count": state.get("retry_count", 0) + 1,
        "logs": [AIMessage(content=f"추가 검색 완료: {new_query}", name="researcher")]
    }

def research_submit_node(state: ResearchState):
    summary_chain = ChatPromptTemplate.from_template(
        "다음 자료를 바탕으로 '{topic}'에 대한 핵심 내용을 요약 정리해줘:\\n\\n{data}"
    ) | llm | StrOutputParser()
    
    final_summary = summary_chain.invoke({"topic": state["topic"], "data": state["raw_data"]})
    
    if "run_id" in state:
        save_step_to_file(state["run_id"], "Research_Done", {"summary": final_summary})
        
    return {"raw_data": final_summary}

research_workflow = StateGraph(ResearchState)
research_workflow.add_node("execute", research_execute_node)
research_workflow.add_node("reflect", research_reflect_node)
research_workflow.add_node("revise", research_revise_node)
research_workflow.add_node("submit", research_submit_node)

research_workflow.add_edge(START, "execute")
research_workflow.add_edge("execute", "reflect")

def route_research(state: ResearchState):
    if state["quality"] == "FAIL" and state.get("retry_count", 0) < 1:
        return "revise"
    return "submit"

research_workflow.add_conditional_edges("reflect", route_research, {"submit": "submit", "revise": "revise"})
research_workflow.add_edge("revise", "submit")
research_workflow.add_edge("submit", END)
research_app = research_workflow.compile()


# ==========================================
# 2. Writer Subgraph
# ==========================================
class WriterState(TypedDict):
    topic: str
    research_data: str
    draft: str
    critique: str
    score: float
    revision_count: int
    logs: Annotated[List[BaseMessage], add_messages]
    code_data: str 
    design_data: str
    run_id: str

def writer_execute_node(state: WriterState):
    count = state.get('revision_count', 0)
    print(f"[Writer Sub] 글 작성 중... (버전 {count + 1})")
    
    chain = ChatPromptTemplate.from_template(
        """당신은 상황에 맞춰 최적의 글을 쓰는 '전문 수석 에디터'입니다.
        제공된 재료들을 바탕으로 주제 '{topic}'에 가장 적합한 형식의 문서를 작성하세요.
        
        [입력 자료]
        1. 연구 내용: {data}
        2. 코드 예제: {code} (없으면 '없음')
        3. 구조도(Mermaid): {design} (없으면 '없음')
        4. 이전 비평: {critique}
        
        [작성 지침]
        1. 형식 판단: 
           - 코드/구조도가 있다면 '기술 문서'나 '튜토리얼' 형식으로, 
           - 없다면 '에세이', '기획서', '보고서' 등 주제에 맞는 형식으로 작성하세요.
           
        2. 자료 통합 (조건부 삽입):
           - 연구 내용: 글의 논리적 근거로 활용하세요.
           - 코드 예제: 내용이 '없음'이 아니라면, 반드시 마크다운 코드 블록(```python ... ```)**으로 본문의 적절한 위치에 삽입하세요. (억지로 만들지 마세요)
           - 구조도: 내용이 '없음'이 아니라면, 반드시 Mermaid 코드 블록(```mermaid ... ```)**으로 시각화 섹션에 삽입하세요.
           
        3. 스타일:
           - 주제가 학술적이면 전문적으로, 대중적이면 읽기 쉽게 작성하세요.
           - 서론-본론-결론의 완결성 있는 구조를 갖추세요.
        """
    ) | llm | StrOutputParser()
    
    draft = chain.invoke({
        "topic": state["topic"],
        "data": state.get("research_data", "자료 없음"),
        "code": state.get("code_data", "없음"), 
        "design": state.get("design_data", "없음"), 
        "critique": state.get("critique", "없음")
    })
    
    if "run_id" in state:
        save_step_to_file(state["run_id"], "Write_Done", {"final_draft": draft})
        
    return {
        "draft": draft, 
        "revision_count": count + 1,
        "logs": [AIMessage(content=f"초안 v{count+1} 작성 완료", name="writer")]
    }

def writer_reflect_node(state: WriterState):
    print("[Writer Sub] 품질 평가 중...")
    
    chain = ChatPromptTemplate.from_template(
        """당신은 세계적인 저널의 '엄격한 수석 편집자'입니다. 
        아래 글이 사용자 요청 주제인 '{topic}'에 완벽하게 부합하는지 비판적으로 평가하세요.
        
        [평가 기준]
        1. 주제 적합성: 요청한 주제를 정확히 다루고 있는가?
        2. 구체성: 막연한 내용이 아니라 구체적인 사실/예시가 있는가?
        3. 논리적 흐름: 서론-본론-결론의 구조가 탄탄한가?
        
        주의: 조금이라도 모호하거나, 평범한 내용이라면 7점 미만으로 점수를 주세요. 
        완벽하지 않으면 9점 이상을 주지 마세요.
        
        형식: 점수/구체적인_피드백 (예: 6.5/주제와 관련 없는 내용이 포함되어 있고 예시가 부족합니다)
        
        [글]: {draft}
        """
    ) | llm | StrOutputParser()
    
    response = chain.invoke({
        "draft": state["draft"],
        "topic": state["topic"] 
    })
    
    try:
        score_str, fb = response.split("/", 1)
        score = float(score_str.strip().replace("점", ""))
    except:
        score, fb = 5.0, "형식 오류"
        
    print(f"      ㄴ 점수: {score}점")
    
    return {
        "score": score, 
        "critique": fb,
        "logs": [AIMessage(content=f"평가: {score}점 / {fb}", name="critic")]
    }
    
writer_workflow = StateGraph(WriterState)
writer_workflow.add_node("execute", writer_execute_node)
writer_workflow.add_node("reflect", writer_reflect_node)
writer_workflow.add_edge(START, "execute")
writer_workflow.add_edge("execute", "reflect")

def route_writer(state: WriterState):
    if state["score"] >= 8.5 or state["revision_count"] >= 3:
        return "end"
    return "execute"

writer_workflow.add_conditional_edges("reflect", route_writer, {"execute": "execute", "end": END})
writer_app = writer_workflow.compile()


# ==========================================
# 3. Code Subgraph
# ==========================================
class CodeState(TypedDict):
    topic: str
    logs: Annotated[List[BaseMessage], add_messages]
    code_result: str
    critique: str
    quality: str
    retry_count: int
    run_id: str

def code_execute_node(state: CodeState):
    print(f"[Code Agent] '{state['topic']}' 코드 초안 작성 중...")
    
    chain = ChatPromptTemplate.from_template(
        """당신은 Senior Python 개발자입니다. 
        주제 '{topic}'에 대한 Python 예제 코드를 작성하세요.
        
        [요구사항]
        1. 실행 가능한 Python 코드여야 합니다.
        2. 코드 내에 상세한 주석(Comments)을 포함하세요.
        3. 마크다운 코드 블록(```python ... ```)으로 감싸지 말고 순수 코드만 출력하거나, 
           코드 블록을 쓴다면 파싱 가능한 형태로 주세요.
        """
    ) | llm | StrOutputParser()
    
    code = chain.invoke({"topic": state["topic"]})
    
    if "run_id" in state:
        save_step_to_file(state["run_id"], "Code_Done", {"code": code})
        
    return {
        "code_result": code, 
        "retry_count": 0,
        "logs": [AIMessage(content="코드 초안 생성 완료", name="coder")]
    }

def code_reflect_node(state: CodeState):
    print("[Code Agent] 코드 품질 리뷰 중...")
    
    chain = ChatPromptTemplate.from_template(
        """당신은 까다로운 코드 리뷰어(Code Reviewer)입니다.
        아래 코드를 검토하고 점수와 피드백을 제공하세요.
        
        [검토할 코드]
        {code}
        
        [평가 기준]
        1. 문법 오류(Syntax Error)가 없는가?
        2. 주석(Comments)이 충분히 작성되었는가?
        3. 실행 가능한 구조인가?
        
        [출력 형식]
        반드시 아래 형식으로만 답변하세요:
        상태: [PASS 또는 FAIL]
        피드백: [구체적인 개선점 또는 오류 내용]
        """
    ) | llm | StrOutputParser()
    
    review_result = chain.invoke({"code": state["code_result"]})
    
    try:
        status_line = review_result.split("\\n")[0]
        quality = "PASS" if "PASS" in status_line else "FAIL"
        critique = review_result
    except:
        quality = "FAIL"
        critique = "리뷰 형식 오류 발생"

    print(f"      ㄴ 리뷰 결과: {quality}")
    return {
        "quality": quality, 
        "critique": critique,
        "logs": [AIMessage(content=f"리뷰 완료: {quality}", name="reviewer")]
    }

def code_revise_node(state: CodeState):
    print(" [Code Agent] 피드백 반영하여 코드 수정 중...")
    
    chain = ChatPromptTemplate.from_template(
        """당신은 개발자입니다. 리뷰어의 피드백을 반영하여 코드를 수정하세요.
        
        [기존 코드]
        {code}
        
        [리뷰어 피드백]
        {critique}
        
        피드백을 반영하여 개선된 '전체 코드'만 다시 출력하세요. (설명 제외)
        """
    ) | llm | StrOutputParser()
    
    new_code = chain.invoke({
        "code": state["code_result"],
        "critique": state["critique"]
    })
    
    return {
        "code_result": new_code,
        "retry_count": state["retry_count"] + 1,
        "logs": [AIMessage(content=f"코드 수정 완료 (시도 {state['retry_count']+1}회)", name="coder")]
    }

code_workflow = StateGraph(CodeState)
code_workflow.add_node("execute", code_execute_node)
code_workflow.add_node("reflect", code_reflect_node)
code_workflow.add_node("revise", code_revise_node)

code_workflow.add_edge(START, "execute")
code_workflow.add_edge("execute", "reflect")

def route_code(state: CodeState):
    if state["quality"] == "PASS" or state["retry_count"] >= 3:
        return END
    return "revise"

code_workflow.add_conditional_edges("reflect", route_code, {"revise": "revise", END: END})
code_workflow.add_edge("revise", "reflect")
code_app = code_workflow.compile()


# ==========================================
# 4. Designer Subgraph
# ==========================================
class DesignerState(TypedDict):
    topic: str
    logs: Annotated[List[BaseMessage], add_messages]
    design_result: str
    critique: str
    quality: str
    retry_count: int
    run_id: str

def designer_execute_node(state: DesignerState):
    print(f"[Designer Agent] '{state['topic']}' 시각화 구조 설계 중...")
    
    chain = ChatPromptTemplate.from_template(
        """당신은 시스템 아키텍트입니다. 
        주제 '{topic}'의 구조나 흐름을 가장 잘 설명할 수 있는 'Mermaid 다이어그램' 코드를 작성하세요.
        
        [요구사항]
        1. 흐름도(graph TD) 또는 시퀀스 다이어그램(sequenceDiagram) 중 적절한 것을 선택하세요.
        2. 설명 텍스트 없이 오직 Mermaid 코드만 출력하세요.
        3. 마크다운 태그(```mermaid)는 제외하고 순수 코드만 주세요.
        """
    ) | llm | StrOutputParser()
    
    design = chain.invoke({"topic": state["topic"]})
    
    if "run_id" in state:
        save_step_to_file(state["run_id"], "Design_Done", {"design": design})
        
    return {
        "design_result": design,
        "retry_count": 0,
        "logs": [AIMessage(content="다이어그램 초안 생성 완료", name="designer")]
    }

def designer_reflect_node(state: DesignerState):
    print("[Designer Agent] 다이어그램 문법 및 적절성 검사 중...")
    
    chain = ChatPromptTemplate.from_template(
        """당신은 Mermaid 문법 전문가입니다. 
        아래 코드가 문법적으로 올바르고 주제를 잘 표현하는지 검사하세요.
        
        [검토할 코드]
        {code}
        
        [출력 형식]
        반드시 아래 형식으로만 답변하세요:
        상태: [PASS 또는 FAIL]
        피드백: [오류 내용 또는 개선점]
        """
    ) | llm | StrOutputParser()
    
    review_result = chain.invoke({"code": state["design_result"]})
    
    try:
        status_line = review_result.split("\\n")[0]
        quality = "PASS" if "PASS" in status_line else "FAIL"
        critique = review_result
    except:
        quality = "FAIL"
        critique = "형식 오류 발생"
        
    print(f"      ㄴ 검사 결과: {quality}")
    return {
        "quality": quality,
        "critique": critique,
        "logs": [AIMessage(content=f"검사 완료: {quality}", name="reviewer")]
    }

def designer_revise_node(state: DesignerState):
    print("[Designer Agent] 피드백 반영하여 수정 중...")
    
    chain = ChatPromptTemplate.from_template(
        """당신은 디자이너입니다. 피드백을 반영하여 Mermaid 코드를 수정하세요.
        
        [기존 코드]
        {code}
        
        [피드백]
        {critique}
        
        수정된 전체 Mermaid 코드만 출력하세요. (설명 제외)
        """
    ) | llm | StrOutputParser()
    
    new_design = chain.invoke({
        "code": state["design_result"],
        "critique": state["critique"]
    })
    
    return {
        "design_result": new_design,
        "retry_count": state["retry_count"] + 1,
        "logs": [AIMessage(content=f"수정 완료 (시도 {state['retry_count']+1}회)", name="designer")]
    }

designer_workflow = StateGraph(DesignerState)
designer_workflow.add_node("execute", designer_execute_node)
designer_workflow.add_node("reflect", designer_reflect_node)
designer_workflow.add_node("revise", designer_revise_node)

designer_workflow.add_edge(START, "execute")
designer_workflow.add_edge("execute", "reflect")

def route_design(state: DesignerState):
    if state["quality"] == "PASS" or state["retry_count"] >= 3:
        return END
    return "revise"

designer_workflow.add_conditional_edges("reflect", route_design, {"revise": "revise", END: END})
designer_workflow.add_edge("revise", "reflect")
designer_app = designer_workflow.compile()


# ==========================================
# 5. Main Supervisor Graph
# ==========================================
def update_agent_results(existing: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    if existing is None:
        return new_data
    merged = existing.copy()
    merged.update(new_data)
    return merged

class MainState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    agent_results: Annotated[Dict[str, Any], update_agent_results]
    next: List[str]
    run_id: str # Pass run_id

class SupervisorDecision(BaseModel):
    next: List[Literal['research_subgraph', 'code_subgraph', 'designer_subgraph', 'writer_subgraph', 'FINISH']] = Field(
        description="다음에 실행할 에이전트 목록. 병렬 실행이 필요하면 여러 개를 선택하세요."
    )
    reasoning: str = Field(description="이 결정을 내린 이유 (성찰)")

def supervisor_node(state: MainState):
    results = state.get("agent_results", {})
    messages = state.get("messages", [])
    last_user_msg = messages[-1].content if messages else ""
    
    status = {
        "research": "있음" if "research" in results else "없음",
        "code": "있음" if "code" in results else "없음",
        "design": "있음" if "design" in results else "없음",
        "final_doc": "있음" if "final_doc" in results else "없음"
    }
    
    system_prompt = f"""당신은 유능한 AI 프로젝트 매니저입니다. 
    사용자의 요청과 현재 작업 상태를 분석하여 최적의 작업자(들)을 지정하세요.
    
    [사용자 요청]: "{last_user_msg}"
    
    [현재 데이터 상태]
    {status}
    
    [판단 가이드]
    1. 코드/디자인 필요성 판단:
           - 요청이 '구현', '개발', '설계', '알고리즘', '구조도' 등을 포함하나요? -> Code/Designer 호출
           - 단순 '동향 파악', '분석 보고서', '에세이'인가요? -> Research만 호출 (Code/Design 생략)
    
    2. 작업 순서:
           1. **0순위: 무조건 종료**:
           - 'Final Document' 상태가 '있음'이라면, 다른 조건 볼 것 없이 무조건 'FINISH'를 선택하세요. (절대 Writer를 다시 부르지 마세요)
    
    2. **중복 실행 금지**: 
           - 이미 'Code 결과'가 '있음'이라면, 절대로 'code_subgraph'를 다시 호출하지 마세요.
           - 이미 'Research 결과'가 '있음'이라면, 절대로 'research_subgraph'를 다시 호출하지 마세요.
    
    3. **작업 흐름**:
           - (1단계) 자료 생성: 요청에 따라 Research, Code, Design 팀을 호출합니다.
           - (2단계) 문서 작성: 위 자료들이 준비되었고, 아직 'final_doc'가 '없음'이라면 -> 'writer_subgraph'를 호출하세요.
       
    4. **특수 상황**:
           - 만약 사용자가 코드만 요청했고 'Code 결과'는 있는데 'Final Document'가 없다면 -> 'writer_subgraph'를 호출하세요.
    3. 병렬 실행:
           - 연구, 코드, 디자인이 모두 필요하다고 판단되면 동시에 호출하세요.
    """
    
    
    print(f"\\n[Main Supervisor] 현재 상태: {status}")

    model = llm.with_structured_output(SupervisorDecision)
    decision = model.invoke([SystemMessage(content=system_prompt)])
    
    # 🛑 Safeguard: If Research is done but LLM selects Research again -> Redirect to Writer
    if "research_subgraph" in decision.next and status["research"] == "있음":
        print("⚠️ [Override] Research already done. Switching to Writer.")
        decision.next = ["writer_subgraph"]

    print(f"\\n[Main Supervisor] 지시: {decision.next}")
    return {"next": decision.next}

def call_research_subgraph(state: MainState):
    print("[Main] 'Research 서브그래프' 호출")
    topic = state["messages"][0].content
    output = research_app.invoke({"topic": topic, "run_id": state.get("run_id","")})
    return {"agent_results": {"research": output["raw_data"]}}

def call_writer_subgraph(state: MainState):
    print("\\n[Main] 'Writer 서브그래프' 호출")
    topic = state["messages"][0].content
    results = state["agent_results"]
    
    output = writer_app.invoke({
        "topic": topic, 
        "research_data": results.get("research", ""),
        "code_data": results.get("code", ""),
        "design_data": results.get("design", ""),
        "revision_count": 0,
        "run_id": state.get("run_id","")
    })
    
    return {"agent_results": {"final_doc": output["draft"]}}

def call_code_subgraph(state: MainState):
    print("[Main] 'Code 팀' (서브그래프) 호출")
    topic = state["messages"][0].content
    output = code_app.invoke({
        "topic": topic,
        "retry_count": 0,
        "run_id": state.get("run_id","")
    })
    return {"agent_results": {"code": output["code_result"]}}

def call_designer_subgraph(state: MainState):
    print("[Main] 'Designer 팀' (서브그래프) 호출")
    topic = state["messages"][0].content
    output = designer_app.invoke({
        "topic": topic,
        "retry_count": 0,
        "run_id": state.get("run_id","")
    })
    return {"agent_results": {"design": output["design_result"]}}

main_workflow = StateGraph(MainState)
main_workflow.add_node("supervisor", supervisor_node)
main_workflow.add_node("research_subgraph", call_research_subgraph)
main_workflow.add_node("writer_subgraph", call_writer_subgraph)
main_workflow.add_node("code_subgraph", call_code_subgraph)
main_workflow.add_node("designer_subgraph", call_designer_subgraph)

main_workflow.add_edge(START, "supervisor")
main_workflow.add_edge("research_subgraph", "supervisor")
main_workflow.add_edge("writer_subgraph", "supervisor")
main_workflow.add_edge("code_subgraph", "supervisor")
main_workflow.add_edge("designer_subgraph", "supervisor")

def route_supervisor(state: MainState):
    next_agents = state["next"]
    if "FINISH" in next_agents:
        return END
    return next_agents

main_workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "research_subgraph": "research_subgraph",
        "code_subgraph": "code_subgraph",
        "designer_subgraph": "designer_subgraph",
        "writer_subgraph": "writer_subgraph",
        END: END
    }
)

memory = MemorySaver()
app = main_workflow.compile(checkpointer=memory)
