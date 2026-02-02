from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

from tools.tavily_search import TripSearchTool

class GuideOutput(BaseModel):
    speaking_expressions: List[str] = Field(description="여행자가 말할 5가지 핵심 표현 (타겟 언어 - 발음 - 한국어 의미)")
    listening_expressions: List[str] = Field(description="여행자가 들을 5가지 핵심 표현 (타겟 언어 - 발음 - 한국어 의미)")
    focused_vocabulary: List[str] = Field(description="해당 장소/상황의 주요 단어 및 추천 항목 (메뉴 포함) 5~7개")
    conversation_flow: List[str] = Field(description="표준 대화 흐름 (단계별)")

import asyncio

async def generate_guide(location: str, situation: str):
    # 1. 데이터 수집 (비동기 병렬 처리 준비)
    search_tool = TripSearchTool()
    search_query = f"{location} {situation} essential phrases, menu, and tips"
    
    # 향후 유튜브 자막 등 다른 비동기 작업과 함께 실행 가능
    # search_task = search_tool.search_place_async(search_query)
    # youtube_task = fetch_youtube_async(...)
    # results = await asyncio.gather(search_task, youtube_task)
    
    # 현재는 검색만 비동기로 실행
    search_result = await search_tool.search_place_async(search_query)
    
    context_text = f"""
    Search Summary: {search_result.get('text_summary', '')}
    Top Results: {str(search_result.get('results', [])[:2])}
    """
    
    # 3. 가이드 생성 (LLM 호출도 비동기로 변경 가능하지만, 현재 체인은 동기 invoke를 많이 사용함.
    # 하지만 LangChain은 ainvoke를 지원하므로 최대한 비동기로 전환)
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    parser = JsonOutputParser(pydantic_object=GuideOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert travel guide creator for Korean travelers."),
        ("user", """
        Location: {location}
        Situation: {situation}
        Context Info: {context}
        
        1. **Determine the Local Language** of `{location}` (e.g., English for US, Japanese for Japan). This is the **Target Language**.
        2. **Output Format**: All expressions and conversation steps MUST follow this format:
           `[Target Language Text] - ([Korean Pronunciation]) - [Korean Meaning]`
           Example: `I'd like a coffee. - (아이드 라이크 어 커피.) - 커피 한 잔 주세요.`
        
        3. **Standard Conversation Flow**:
           - Create a realistic dialogue script.
           - Format: `Step N: [Actor] [Target Language Text] - ([Pronunciation]) - ([Meaning])`
           
        4. **Focused Vocabulary**:
           - Include key terms, menu items, or signs relevant to the `{location}` and `{situation}`.
           - Combine recommended menu items here if applicable.
        
        {format_instructions}
        """)
    ])
    
    chain = prompt | llm | parser
    
    try:
        # 비동기 LLM 호출
        guide = await chain.ainvoke({
            "location": location,
            "situation": situation,
            "context": context_text,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        # 검색 실패 또는 키 누락 시 대체
        guide = {
            "speaking_expressions": ["I'd like... - (Pronunciation) - Meaning"],
            "listening_expressions": ["For here or to go? - (Pronunciation) - Meaning"],
            "focused_vocabulary": ["Cilantro (고수)", "Spicy (매운)", "To go (포장)"],
            "conversation_flow": [
                "Step 1: [Staff] Hello - (Hello) - (안녕하세요)",
                "Step 2: [Traveler] Hi - (Hi) - (안녕)"
            ]
        }
        print(f"Guide generation error: {e}")

    # 채팅 에이전트를 위한 컨텍스트 병합
    full_context = {
        "guide": guide,
        "raw_search": search_result,
        "menu_text": str(guide.get("focused_vocabulary", "No vocab data")),
        # 에이전트가 참고할 수 있도록 표현 통합
        "key_phrases": guide.get("speaking_expressions", []) + guide.get("listening_expressions", [])
    }
    
    return full_context
