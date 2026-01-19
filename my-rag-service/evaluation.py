import os
import uuid
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

client = Client()

# --- 1. 평가 데이터셋 생성 ---
dataset_name = "pipeline_evaluation_week10"

def create_evaluation_dataset():
    if client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset '{dataset_name}' already exists.")
        return client.read_dataset(dataset_name=dataset_name)
    
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="RAG Pipeline Quality Evaluation Dataset",
    )
    
    # 예제 데이터 추가
    test_cases = [
        {
            "input": {"messages": [{"role": "user", "content": "LangGraph에 대해 설명해줘"}]},
            "expected": {"has_summary": True, "key_topics": ["StateGraph", "Node", "Edge"]}
        },
        {
            "input": {"messages": [{"role": "user", "content": "Python으로 피보나치 수열 코드 짜줘"}]},
            "expected": {"has_code": True, "language": "python"}
        }
    ]
    
    client.create_examples(
        inputs=[case["input"] for case in test_cases],
        outputs=[case["expected"] for case in test_cases],
        dataset_id=dataset.id,
    )
    print(f"Created dataset '{dataset_name}' with {len(test_cases)} examples.")
    return dataset


# --- 2. 자동 평가 함수 (LLM-as-a-Judge) ---

evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def evaluate_pipeline_output(run, example):
    """
    run: LangSmith의 실행 객체 (출력값 포함)
    example: 데이터셋의 예제 객체 (입력값 및 정답 출력값 포함)
    """
    # 실제 출력값 및 예상 정답 추출
    # 참고: 파이프라인의 반환 구조에 따라 로직 조정이 필요할 수 있음
    
    # run.outputs가 전체 상태(State)일 수 있으므로 'agent_results'를 추출
    outputs = run.outputs if run.outputs else {}
    agent_results = outputs.get("agent_results", {})
    
    # 최종 텍스트 추출 시도
    actual_text = ""
    if "final_doc" in agent_results:
        actual_text = agent_results["final_doc"]
    elif "research" in agent_results:
        actual_text = agent_results["research"]
    elif "code" in agent_results:
        actual_text = agent_results["code"]
    
    expected = example.outputs if example.outputs else {}
    input_text = example.inputs["messages"][0]["content"]

    # LLM 심판(Judge) 프롬프트
    prompt = ChatPromptTemplate.from_template("""
    You are an expert evaluator for RAG systems.
    
    [Input Question]: {input}
    [Actual Output]: {actual}
    [Expected Criteria]: {expected}
    
    Evaluate the Output based on the following metrics:
    1. Completeness (0-1): Does it answer the question fully?
    2. Relevance (0-1): Is it relevant to the input?
    3. Hallucination (0-1): Does it contain non-factual info? (1 = Hallucinated, 0 = Clean)
    4. Format (0-1): Does it follow requested format (e.g. code blocks)?
    
    Return JSON:
    {{
        "completeness": 0.8,
        "relevance": 0.9,
        "hallucination": 0.0,
        "format": 1.0,
        "reason": "Brief explanation"
    }}
    """)
    
    chain = prompt | evaluator_llm | JsonOutputParser()
    
    try:
        score_data = chain.invoke({
            "input": input_text,
            "actual": actual_text,
            "expected": str(expected)
        })
    except Exception as e:
        print(f"Evaluation failed: {e}")
        score_data = {"completeness": 0, "relevance": 0, "hallucination": 0, "format": 0, "reason": "Error"}

    return {
        "key": "quality_metrics",
        "score": score_data["completeness"], # 주요 점수
        "comment": score_data["reason"],
        # 추가 지표는 필요 시 별도로 로깅하거나, 딕셔너리 형태로 반환 가능
    }

# --- Main Execution ---
if __name__ == "__main__":
    create_evaluation_dataset()
    print("Evaluation script ready. Use 'langsmith evaluate' or integration tests to run it.")
