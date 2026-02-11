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

from database.supabase_client import GuideCache

# 전역 캐시 인스턴스
guide_cache = GuideCache()

from langchain_community.document_loaders import YoutubeLoader
import re
import asyncio

async def fetch_youtube_context(query: str) -> str:
    """
    유튜브 검색(Tavily 경유) 후 자막을 추출하여 반환합니다.
    LangChain YoutubeLoader를 사용하여 자막 처리를 간소화합니다.
    """
    try:
        search_tool = TripSearchTool()
        # "site:youtube.com"을 붙여 유튜브 영상 위주로 검색
        search_result = await search_tool.search_place_async(f"{query} site:youtube.com")
        results = search_result.get("results", [])
        
        video_ids = []
        for res in results:
            url = res.get("url", "")
            # 유튜브 Video ID 추출 (v=값)
            match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
            if match:
                video_ids.append(match.group(1))
        
        # 중복 제거 및 최대 2개만 사용
        video_ids = list(set(video_ids))[:2]
        
        full_transcript = ""
        loop = asyncio.get_running_loop()
        
        for vid in video_ids:
            try:
                # YoutubeLoader 초기화 (한국어 -> 영어 순)
                loader = YoutubeLoader.from_youtube_url(
                    f"https://www.youtube.com/watch?v={vid}",
                    add_video_info=False,
                    language=["ko", "en"]
                )
                
                # 동기 함수인 load()를 비동기 루프로 실행하여 블로킹 방지
                docs = await loop.run_in_executor(None, loader.load)
                
                # 텍스트 추출
                text = " ".join([d.page_content for d in docs])
                full_transcript += f"\n[Video {vid}]: {text[:1000]}..."
                
            except Exception as e:
                # 자막이 없거나 로드 실패 시 무시
                continue
                
        return full_transcript if full_transcript else "No YouTube transcripts found."
        
    except Exception as e:
        print(f"YouTube Search Error: {e}")
        return "YouTube search failed."

class SearchQuery(BaseModel):
    specific_query: str = Field(description="Web search query for specific info (menu, price, tips)")
    general_query: str = Field(description="YouTube search query for broad context (brand name + ordering guide/vlog)")

async def generate_guide(location: str, situation: str):
    # 0. 캐시 확인 (0.5초 컷)
    cached_guide = await guide_cache.search_guide(location, situation)
    if cached_guide:
        return {
            "guide": cached_guide,
            "raw_search": {"summary": "Cached Data"},
            "menu_text": str(cached_guide.get("focused_vocabulary", "Cached vocab")),
            "key_phrases": cached_guide.get("speaking_expressions", []) + cached_guide.get("listening_expressions", [])
        }

    # 1. 검색어 최적화 (Query Refinement) - LLM 사용
    # 사용자 입력: "도쿄 디즈니 입구 근처 편의점", "물이랑 간식 사기"
    # -> Specific: "tokyo disneyland entrance convenience store snack price"
    # -> General: "Japanese convenience store buying snacks vlog" (브랜드/업종 추출)
    
    refiner_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    refiner_parser = JsonOutputParser(pydantic_object=SearchQuery)
    
    refiner_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a search query optimizer for a travel guide AI."),
        ("user", """
        Location: {location}
        Situation: {situation}
        
        Generate two search queries:
        1. **specific_query** (for Web): Detailed query to find menu, prices, and tips.
        2. **general_query** (for YouTube): Broad query to find vlogs or ordering guides. 
           - Extract the Core Brand Name or Category (e.g., 'Starbucks', 'McDonalds', 'Convenience Store').
           - Append keywords like 'ordering guide', 'vlog', 'how to order'.
           - If the location is specific (e.g., 'Starbucks Shibuya'), use the Brand Name ('Starbucks') for the general query to get more results.
        
        {format_instructions}
        """)
    ])
    
    refiner_chain = refiner_prompt | refiner_llm | refiner_parser
    
    try:
        # 검색어 생성 (빠른 응답을 위해 gpt-5-mini 사용)
        query_result = await refiner_chain.ainvoke({
            "location": location,
            "situation": situation,
            "format_instructions": refiner_parser.get_format_instructions()
        })
        specific_query = query_result.get("specific_query", f"{location} {situation} menu price")
        general_query = query_result.get("general_query", f"{location} ordering vlog")
        
    except Exception as e:
        print(f"Query Refinement Failed: {e}")
        specific_query = f"{location} {situation} menu price tips"
        general_query = f"{location} ordering guide vlog"

    # 2. 하이브리드 검색 (Hybrid Search) - 병렬 실행
    search_tool = TripSearchTool()
    
    print(f"🚀 Starting Hybrid Search...\n- Specific: {specific_query}\n- General: {general_query}")
    
    # Asyncio Gather로 병렬 실행
    tavily_task = search_tool.search_place_async(specific_query)
    youtube_task = fetch_youtube_context(general_query)
    
    results = await asyncio.gather(tavily_task, youtube_task)
    search_result = results[0]  # Tavily 결과
    youtube_context = results[1] # YouTube 자막 결과
    
    context_text = f"""
    [Web Search Result]:
    {search_result.get('text_summary', '')}
    
    [YouTube Vlog Context]:
    {youtube_context}
    """
    
    # 3. 가이드 생성
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    parser = JsonOutputParser(pydantic_object=GuideOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert travel guide creator for Korean travelers."),
        ("user", """
        Location: {location}
        Situation: {situation}
        
        We have gathered information from Web and YouTube Vlogs.
        **Context Info**:
        {context}
        
        1. **Determine the Local Language** of `{location}`.
        2. use **Context Info** to find real expressions, menu items, and ordering tips.
           - If YouTube context contains actual dialogue, prioritize it for "Conversation Flow".
        
        3. **Output Format** (Strictly follow this):
           - Expressions: `[Target Lang] - ([Pronunciation]) - [Meaning]`
           
        4. **Contents**:
           - Speaking/Listening Expressions (5 each)
           - Focused Vocabulary (Menu/Terms)
           - Conversation Flow (Step-by-step dialogue)
        
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
        
        # 4. 캐시 저장 (비동기로 수행하여 사용자 응답 속도 저하 최소화)
        # await save_guide(...) waits here. Ideally use create_task but to ensure save use await.
        await guide_cache.save_guide(location, situation, guide)
        
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
