# ✈️ TripTalker: 실전 여행 회화 시뮬레이터

**TripTalker**는 여행 상황에 맞는 가이드를 생성하고, AI 점원과 실전처럼 대화하며 여행 회화를 연습할 수 있는 시뮬레이터입니다.


TripTalker는 **하이브리드 검색(Hybrid Search)**과 **의미 기반 캐시(Semantic Cache)**를 결합하여 빠르고 정확한 가이드를 제공합니다.



## ✨ 핵심 기술 (Core Features)

1.  **Google Places Autocomplete**:
    -   Google Maps API를 통해 정확한 장소 명칭과 주소를 자동완성으로 제공합니다.

2.  **Hybrid Search Strategy (Async)**:
    -   **Tavily (Web)**: 최신 메뉴, 가격, 현지 팁 등 구체적인 정보를 검색합니다.
    -   **YouTube (Vlog)**: 영상 자막을 분석하여 실제 현지인들이 사용하는 생생한 회화 표현을 추출합니다.
    -   이 두 과정을 `asyncio`로 병렬 처리하여 속도를 최적화했습니다.

3.  **Semantic Caching (Supabase)**:
    -   `pgvector`를 활용하여 질문의 의미(Semantic)를 분석합니다.
    -   유사한 질문(예: "오사카 라면" vs "오사카 라멘")이 있으면 **0.5초** 만에 저장된 가이드를 반환합니다.

4.  **Real-Time Simulation**:
    -   LangGraph 기반의 AI 에이전트(Clerk, Tutor)가 상황에 맞는 페르소나를 연기합니다.
    -   답변은 타자기 효과(Streaming)로 실시간 출력됩니다.

## 🚀 실행 방법

### 1. 환경 설정 (.env)
```ini
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
SUPABASE_URL=...
SUPABASE_KEY=...
GOOGLE_MAPS_API_KEY=...
```

### 2. 실행
```bash
# 필수 패키지 설치
pip install -r requirements.txt

# 앱 실행
python trip-talk/app.py
```
