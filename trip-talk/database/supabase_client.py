import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 환경 변수 로드
load_dotenv()

class GuideCache:
    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        self.enabled = bool(self.supabase_url and self.supabase_key)
        
        if self.enabled:
            print("✅ Supabase Cache Enabled")
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # 테이블 이름이 'documents'이고 query_name이 'match_documents'인 것으로 가정 (LangChain 기본값)
            # 사용자가 Supabase SQL Editor에서 해당 테이블과 함수를 생성해야 함.
            self.vector_store = SupabaseVectorStore(
                client=self.client,
                embedding=self.embeddings,
                table_name="documents",
                query_name="match_documents"
            )
        else:
            print("⚠️ Supabase Credentials missing. Caching is DISABLED.")

    async def search_guide(self, location: str, situation: str, threshold: float = 0.78):
        """
        주어진 장소와 상황에 대한 가이드가 캐시에 있는지 검색합니다.
        유사도가 threshold 이상인 경우 결과를 반환합니다.
        (0.9 -> 0.78 로 완화: '오사카 라멘' vs '오사카 라면' 정도의 차이를 허용하기 위함)
        """
        if not self.enabled:
            return None
            
        query_text = f"Location: {location}, Situation: {situation}"
        print(f"🔍 Searching cache for: {query_text}...")
        
        try:
            # LangChain의 similarity_search_with_relevance_scores 사용
            # Note: SupabaseVectorStore implementation might vary, ensuring synchronous call works or wrapping it if needed.
            # Most vector stores in LangChain are synchronous. 
            # We run this potentially blocking call. In a full async app, we might want run_in_executor.
            
            # LangChain 대신 직접 RPC 호출 (호환성 문제 해결)
            query_embedding = self.embeddings.embed_query(query_text)
            
            params = {
                "query_embedding": query_embedding,
                "match_threshold": threshold, # 0.78 etc.
                "match_count": 1
            }
            
            # 직접 RPC 호출
            response = self.client.rpc("match_documents", params).execute()
            
            # Supabase Python v2+ response format: response.data
            results = response.data
            
            if not results:
                print("Cache Miss (No results)")
                return None
                
            # 결과: [{'id':..., 'content':..., 'metadata':..., 'similarity':...}]
            best_match = results[0]
            score = best_match.get("similarity", 0)
            print(f"Cache Score: {score}")
            
            if score >= threshold:
                print("⚡ Cache HIT!")
                return best_match.get("metadata", {}).get("guide_json")
            else:
                print("Cache Miss (Low similarity)")
                return None
                
        except Exception as e:
            print(f"Cache Search Error: {e}")
            return None

    async def save_guide(self, location: str, situation: str, guide_data: dict):
        """
        생성된 가이드를 Supabase에 저장합니다.
        """
        if not self.enabled:
            return
            
        text_content = f"Location: {location}, Situation: {situation}"
        embedding = self.embeddings.embed_query(text_content)
        
        metadata = {
            "guide_json": guide_data,
            "location": location,
            "situation": situation
        }
        
        row = {
            "content": text_content,
            "metadata": metadata,
            "embedding": embedding
        }
        
        try:
            print("💾 Saving to cache...")
            # 직접 Insert 호출
            self.client.table("documents").insert(row).execute()
            print("✅ Saved to cache.")
        except Exception as e:
            print(f"Cache Save Error: {e}")
