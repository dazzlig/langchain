from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
import os

class TripSearchTool:
    def __init__(self, k=5):
        # 1. API 래퍼는 기본 설정만
        self.search_wrapper = TavilySearchAPIWrapper(
            tavily_api_key=os.getenv("TAVILY_API_KEY")
        )
        
        # 2. 도구 설정에 옵션 포함
        self.tool = TavilySearchResults(
            api_wrapper=self.search_wrapper,
            max_results=k,
            include_images=True,
            include_answer=True,
            include_raw_content=True,
            search_depth="advanced"
        )

    def search_place(self, query: str):
        try:
            results = self.tool.invoke({"query": query})
            
            # 결과 아이템에 포함된 이미지 URL들을 수집합니다.
            all_images = []
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and "images" in item:
                        all_images.extend(item.get("images", []))
            
            return {
                "text_summary": "", # 도구 반환값 리스트에는 answer가 별도로 분리되어 있지 않을 수 있음
                "results": results,
                "images": all_images
            }
        except Exception as e:
            return {"error": str(e), "results": [], "images": []}

    async def search_place_async(self, query: str):
        try:
            # 비동기 실행을 위해 ainvoke 사용
            results = await self.tool.ainvoke({"query": query})
            
            all_images = []
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and "images" in item:
                        all_images.extend(item.get("images", []))
            
            return {
                "text_summary": "", 
                "results": results,
                "images": all_images
            }
        except Exception as e:
            return {"error": str(e), "results": [], "images": []}
