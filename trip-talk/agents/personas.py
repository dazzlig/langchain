from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import TripTalkerState

def clerk_node(state: TripTalkerState):
    messages = state["messages"]
    context = state.get("context_data", {})
    location = state.get("location", "Unknown Place")
    situation = state.get("situation", "Unknown Situation")
    
    # 컨텍스트를 사용하여 시스템 프롬프트 구성
    guide = context.get("guide", {})
    conversation_flow = guide.get("conversation_flow", [])
    flow_str = "\n".join(conversation_flow) if conversation_flow else "Standard service flow"
    
    context_str = f"""
    Menu/Info: {context.get('menu_text', 'No menu info')}
    Key Phrases: {context.get('key_phrases', [])}
    Standard Conversation Flow:
    {flow_str}
    """
    
    system_prompt = f"""You are a clerk/staff member at {location}.
    The situation is: {situation}.
    
    Your goal is to simulate a realistic interaction with a traveler who is learning the language.
    
    **CRITICAL INSTRUCTIONS**:
    1. **REALISM & CONTEXT**: Base your questions and options on the actual "Menu/Info" provided. Be a real clerk in that specific location.
    2. **NATURAL INTERACTION**: Do NOT ask a long list of questions at once (e.g., "Rice? Beans? Salsa? Drink?"). Ask ONE thing at a time, just like a real person.
    3. **NO ROBOTIC SCRIPTS**: You do not need to strictly follow the "Standard Conversation Flow" if the user guides the conversation differently. React naturally.
    4. **CONSISTENT STYLE**: Keep your responses concise (1-2 sentences).
    
    Context Info:
    {context_str}
    """
    
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    return {"messages": [response]}

def tutor_node(state: TripTalkerState):
    messages = state["messages"]
    context = state.get("context_data", {})
    
    system_prompt = """You are a helpful language tutor copilot.
    The user is practicing a travel scenario.
    They asked a question or need help.
    Provide clear explanations, suggest natural expressions, and guide them back to the role-play.
    
    Use the context if relevant (e.g., explaining items on the menu).
    """
    
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.5)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    return {"messages": [response]}
