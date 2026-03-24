import os
import json
from groq import Groq

def build_prompt(employee_data: dict, ml_result: dict) -> str:
    """Build the system prompt for HR strategist."""
    emp = employee_data
    # Use top 5 SHAP factors from grouped features
    shap_items = []
    for i, f in enumerate(ml_result.get('top_factors', [])[:5], 1):
        shap_items.append(f"{i}. {f['feature']}: {f['impact']}")
    shap_formatted = "\n".join(shap_items)

    prompt = f"""You are an expert HR Business Partner and Retention Strategist.

EMPLOYEE CONTEXT:
Employee #: {emp.get('EmployeeNumber', 'N/A')}
Job Role: {emp.get('JobRole', 'N/A')} in {emp.get('Department', 'N/A')}
Job Level: {emp.get('JobLevel', 'N/A')}/5 | Tenure: {emp.get('YearsAtCompany', 'N/A')} years
Monthly Income: ${emp.get('MonthlyIncome', 'N/A')} | Last Hike: {emp.get('PercentSalaryHike', 'N/A')}%
Performance: {emp.get('PerformanceRating', 'N/A')}/4 | Work-Life Balance: {emp.get('WorkLifeBalance', 'N/A')}/4
Overtime: {emp.get('OverTime', 'N/A')} | Distance from Home: {emp.get('DistanceFromHome', 'N/A')} km
Job Satisfaction: {emp.get('JobSatisfaction', 'N/A')}/4 | Environment Satisfaction: {emp.get('EnvironmentSatisfaction', 'N/A')}/4

PREDICTIVE ML DIAGNOSIS:
Current Flight Risk: {ml_result['risk_score']}% ({ml_result['risk_tier']})

THE "WHY" (Mathematical Risk Drivers from SHAP):
The Machine Learning model specifically identified these top factors pushing the risk score up or down.
Positive impacts (+) indicate Attrition Drivers (factors increasing the likelihood of departure). 
Negative impacts (-) indicate Retention Anchors (factors currently maintaining their engagement).
{shap_formatted}

TASK:
Provide a concise, 2-section response. Do not use filler introductions.
Maintain a highly professional, objective HR business tone. Do not use casual phrasing like "red flags", "green flags", or "sticking around".
Use specific numbers from the context above.
CRITICAL: Translate all raw technical variable names (e.g., OverTimeFlag, JobRole, JobSatisfaction) into natural, human-readable business language (e.g., "overtime requirements", "current position", "overall job satisfaction"). Never use raw CamelCase variable names in any part of your response.

1. "risk_narrative": A highly readable, scannable professional summary. Use Markdown **bolding** for important metrics and key concepts. Make sentences short and punchy. You MUST explicitly explain:
   (a) The primary Attrition Drivers elevating this employee's departure risk.
   (b) The key Retention Anchors currently maintaining their engagement (if any negative SHAP values are present).

2. "retention_strategies": A bulleted list of exactly 3 highly specific,
   actionable, and cost-effective managerial interventions designed to mitigate the identified Attrition Drivers. Use Markdown **bolding** for emphasis if needed.

3. "suggested_questions": An array of exactly 3 ULTRA-SHORT (maximum 8 words each) follow-up questions 
   the manager should ask you (the AI consultant) next. These questions should be 
   highly specific to this employee's unique Attrition Drivers. Prevent generic questions.
"""
    return prompt

def generate_insights(employee_data: dict, ml_result: dict) -> dict:
    """
    Call Groq LLM to generate risk narrative and retention strategies.

    Returns a dict with keys: risk_narrative, retention_strategies
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    prompt = build_prompt(employee_data, ml_result)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert HR strategist. You MUST respond ONLY with valid JSON. Do not include markdown blocks or conversational text. The JSON must have exactly these keys: risk_narrative (string), retention_strategies (array of 3 strings), suggested_questions (array of 3 strings)."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=600
    )
    content = response.choices[0].message.content.strip()
    
    # Clean up standard markdown wrapping if the LLM adds it
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    try:
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
            return json.loads(content, strict=False)
        else:
            raise ValueError("No JSON block found")
    except Exception as e:
        print(f"JSON Decoding Error from Groq: {e}\nContent was: {content}")
        return {
            "risk_narrative": "The AI model encountered a temporary formatting issue (unescaped quotes). Please click 'Run ML Inference' again to regenerate the diagnostic summary.",
            "retention_strategies": [
                "Please rerun inference to view tailored retention strategies.",
                "Review the Mathematical Risk Drivers (SHAP) chart for immediate variable impact insights.",
                "If this error persists on this exact row, try modifying a single value slightly."
            ],
            "suggested_questions": []
        }

def chat_with_consultant(employee_data: dict, ml_analysis: dict, messages: list) -> dict:
    """Takes the history and generates the next conversational response as a JSON dict."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    
    # Format the context similar to earlier
    # We don't need to rebuild context, just the system prompt

    system_content = f"""
You are an expert HR Strategic Consultant directly advising a manager.
CONTEXT:
Employee Data: {json.dumps(employee_data)}
ML Analysis (SHAP): {json.dumps(ml_analysis)}

Your goal is to answer the manager's follow-up questions concisely, professionally, and actionably.
Maintain a highly professional, objective HR business tone. Do not use casual phrasing.
CRITICAL INSTRUCTIONS:
1. You MUST respond ONLY with valid JSON. Do not include markdown blocks or conversational text outside the JSON.
2. The JSON must have exactly these keys: "reply" (string) and "suggested_questions" (array of exactly 3 strings).
3. "reply": Your actual chat response. You may use bullet points and Markdown **bolding** for emphasis. You MUST escape all newlines as \\n. Do not use literal physical line breaks inside the JSON string.
4. "suggested_questions": Exactly 3 ULTRA-SHORT (maximum 8 words each) follow-up questions the manager should ask you next.
"""

    formatted_messages = [{"role": "system", "content": system_content}]
    
    # Append history
    for msg in messages[-5:]:
        # Handle both dicts and Pydantic models gracefully
        role = msg.role if hasattr(msg, 'role') else msg.get('role', 'user')
        content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
        formatted_messages.append({"role": role, "content": content})
        
    response = client.chat.completions.create(
        model=model,
        messages=formatted_messages,
        temperature=0.4,
        max_tokens=500
    )
    content = response.choices[0].message.content.strip()
    
    # Clean up standard markdown wrapping if the LLM adds it
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    try:
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
            return json.loads(content, strict=False)
        else:
            return {"reply": "I'm having a bit of trouble formulating that response mathematically. Can we try rephrasing the question?", "suggested_questions": []}
    except Exception as e:
        print(f"Chat JSON Error: {e} | Content: {content}")
        return {"reply": "I apologize, but I encountered a formatting error while structuring my response. Let's try continuing the discussion.", "suggested_questions": ["What should we focus on next?"]}
