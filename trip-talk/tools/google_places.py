import googlemaps
import os
from typing import List, Dict

class GooglePlacesTool:
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = googlemaps.Client(key=self.api_key)
                print("✅ Google Maps Client Initialized")
            except Exception as e:
                print(f"⚠️ Google Maps Init Error: {e}")
        else:
            print("⚠️ GOOGLE_MAPS_API_KEY not found. Places search disabled.")

    def search_places(self, query: str) -> List[Dict[str, str]]:
        """
        검색어에 대한 장소 자동완성 결과를 반환합니다.
        Return: [{'description': 'Full Text', 'place_id': '...'}]
        """
        if not self.client or not query:
            return []

        try:
            # Places Autocomplete API 호출
            # input_type='textquery' is for find_place, but for autocomplete we use places_autocomplete
            results = self.client.places_autocomplete(
                input_text=query,
                types=['establishment', 'geocode'], # 장소 위주
                language='ko' # 한국어 결과 선호
            )
            
            # 필요한 정보만 추출
            suggestions = []
            for item in results:
                suggestions.append({
                    "description": item.get("description", ""),
                    "place_id": item.get("place_id", ""),
                    "main_text": item.get("structured_formatting", {}).get("main_text", "")
                })
            
            return suggestions
            
        except Exception as e:
            print(f"Google Places Error: {e}")
            return []

# 전역 인스턴스 (필요시 사용)
# places_tool = GooglePlacesTool()
