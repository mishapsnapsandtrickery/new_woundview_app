import openai
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from dotenv import load_dotenv
load_dotenv('apikey.env')  # apikey.env 파일 명시적 로드

def generate_advice_openai(
    predict_result,
    wound_size,
    redness,
    swelling,
    heat,
    pain,
    function_loss,
    date,
    bodypart,
    cause
):
    # 환경변수에서 OpenAI 키 로딩 (직접 문자열로 넣어도 됨)
    

    wound_width = wound_size.get('wound_width') if isinstance(wound_size, dict) else None
    wound_height = wound_size.get('wound_height') if isinstance(wound_size, dict) else None
    wound_area = wound_size.get('wound_area') if isinstance(wound_size, dict) else None

    prompt = f"""너는 상처 분석 전문가로서, 딥러닝 모델이 예측한 상처 정보와 사용자가 체크박스로 입력한 염증 증상(피부가 빨개졌는지, 부어 있는지, 열이 나는지, 아픈지, 움직이기 어려운지)을 바탕으로 다음 항목을 마크다운 형식으로 순차적으로 출력해줘.
    [입력 항목]
    - 상처의 발생일자: {date}
    - 상처의 발생부위: {bodypart}
    - 상처의 발생원인: {cause}
    
    - 상처 종류: {predict_result}
    - 상처 크기: {wound_width} x {wound_height}cm (면적: {wound_area}㎟)
    - 사용자 입력 염증 증상:
    - 피부가 빨개졌나요? → {redness}
    - 부어 있나요? → {swelling}
    - 만지면 뜨겁게 느껴지나요? → {heat}
    - 아프거나 따끔거리나요? → {pain}
    - 움직이기 어렵거나 사용이 불편한가요? → {function_loss}

    [출력 항목 형식 (JSON으로 작성)]

    **위험등급**:  
    - 저위험 / 중간위험 / 고위험 중 하나로 판단

    **증상 해석**:  
    - 상처 종류와 크기, 입력된 증상을 종합해서 상처 상태를 설명해줘.  
    - 염증 반응과 통증, 기능장애가 실제 상태에 어떤 영향을 주는지 자연스럽게 해석해줘.  
    - 어려운 용어는 괄호 병기로 쉽게 풀어줘 (예: 부종(붓기), 발적(붉어짐)).
    - 상처의 크기에 대한 정보는 출력하지 말아줘.

    **회복 예상 기간**:  
    - 몇 일~몇 주 내 회복될 수 있는지 예측하고, 근거를 간단히 설명해줘.

    **관리 가이드**:  
    - 실생활에서 따라할 수 있는 단계별 상처 관리 방법을 3~5가지로 제시해줘.  
    - 세척, 소독, 드레싱, 연고 사용 등 실용적인 내용을 포함하고, 연고는 성분 위주로 추천해줘 (예: 시카, 판테놀 등).

    **주의사항**:  
    - 감염 예방, 상처 자극 방지, 드레싱 교체 등에 대해 핵심 주의사항 2~4가지를 알려줘.

    **행동 지침**:  
    - 위험도에 따라 병·의원 방문 필요성 여부를 판단해줘.  
    - 고위험 → 병원 또는 응급실 즉시 내원 권장  
    - 중간위험 → 병·의원 진료 권장  
    - 저위험 → 자가 치료 가능, 변화 시 병원 상담

    꼭 JSON 형식으로 반환해줘.
    예시 :
    {{
    "risk_level": "고위험",
    "symptom_analysis": "상처는 베인 상처로, 가로 4cm, 세로 2cm로 비교적 크며, 피부가 붉고 부어 있고, 만지면 뜨거우며 통증이 심하게 나타납니다. 이는 염증 반응이 활발하게 진행 중임을 의미합니다. 부종(붓기), 발적(붉어짐), 열감 등은 감염의 위험 신호일 수 있습니다.",
    "recovery_period": "약 2~3주 내 회복될 수 있으나, 감염이 동반될 경우 더 오래 걸릴 수 있습니다. 상처의 크기와 염증 증상이 심한 점을 고려했습니다.",
    "care_guide": "1. 흐르는 물로 상처를 깨끗이 세척하세요.\n2. 소독제를 사용해 감염을 예방하세요.\n3. 멸균 거즈로 상처를 덮고, 하루 1~2회 드레싱을 교체하세요.\n4. 시카, 판테놀 등 진정 성분의 연고를 사용하세요.\n5. 상처 부위를 자극하지 않도록 주의하세요.",
    "caution": "상처 부위가 더 붓거나, 고름이 생기거나, 열이 지속되면 즉시 병원을 방문하세요. 드레싱은 항상 깨끗한 상태로 유지하고, 손을 깨끗이 씻은 후 처치하세요.",
    "action_guide": "감염 및 합병증 우려가 높으므로 즉시 병원이나 응급실 방문을 권장합니다.",
    }}
    """

    import json
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 의료 전문가이자 상처 분석 도우미야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        try:
            result_json = json.loads(answer)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', answer, re.DOTALL)
            if match:
                result_json = json.loads(match.group())
            else:
                return {"error": "LLM 응답에서 JSON을 추출할 수 없습니다.", "raw": answer}
        return result_json
    except Exception as e:
        return {"error": "AI 서버와 통신에 실패했습니다.", "detail": str(e)}