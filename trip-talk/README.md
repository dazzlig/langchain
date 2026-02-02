# ✈️ TripTalker: 실전 여행 회화 시뮬레이터

**TripTalker**는 여행 상황에 맞는 가이드를 생성하고, AI 점원과 실전처럼 대화하며 여행 회화를 연습할 수 있는 시뮬레이터입니다.

## 🛠️ 시스템 아키텍처 (System Architecture)

```mermaid
graph TD
    %% Define styles
    classDef ui fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef logic fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef ai fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef external fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    subgraph UserInterface [Frontend (Gradio)]
        Input[입력: 장소 & 상황]:::ui
        GuideUI[가이드 출력 화면]:::ui
        ChatUI[실시간 채팅 화면]:::ui
    end

    subgraph Backend [Backend Logic]
        ContextGen[컨텍스트 생성기<br/>(Async Guide Generator)]:::logic
        LangGraph[LangGraph<br/>(State Management)]:::logic
        Router[Router Node]:::logic
    end

    subgraph AI_Agents [AI Persona Agents]
        Clerk[Clerk Agent<br/>(GPT-5-mini)]:::ai
    end

    subgraph ExternalServices [External Services]
        Tavily[Tavily Search API<br/>(Async)]:::external
        OpenAI[OpenAI GPT-5-mini]:::external
    end

    %% Flow: Guide Generation
    Input -->|1. Generate Click| ContextGen
    ContextGen -->|Async Search| Tavily
    ContextGen -->|Context Prompt| OpenAI
    OpenAI -->|Guide Data<br/>(Flow, Expr, Vocab)| GuideUI

    %% Flow: Chat Simulation
    ChatUI -->|2. User Message| LangGraph
    GuideUI -.->|Inject Context| LangGraph
    LangGraph --> Router
    Router --> Clerk
    Clerk -->|System Prompt| OpenAI
    OpenAI -->|Streaming Response| ChatUI
```

## ✨ 주요 기능
1. **맞춤형 가이드 생성**: 장소와 상황만 입력하면 Tavily 검색을 통해 실시간 정보를 반영한 회화 가이드(대화 흐름, 핵심 표현, 단어)를 제공합니다.
2. **실전 대화 시뮬레이션**: LangGraph 기반의 AI 에이전트가 실제 점원처럼 행동하며 사용자와 롤플레잉을 진행합니다.
3. **고성능 아키텍처**:
    - **Async IO**: 검색 및 가이드 생성을 비동기로 처리하여 대기 시간 단축
    - **Streaming**: 채팅 답변을 실시간으로 스트리밍하여 빠른 반응 속도 제공
    - **Long Context**: 복잡한 체인 없이 긴 컨텍스트를 한 번에 처리

## 🚀 실행 방법
```bash
# 필수 패키지 설치
pip install -r requirements.txt

# 앱 실행
python app.py
```
