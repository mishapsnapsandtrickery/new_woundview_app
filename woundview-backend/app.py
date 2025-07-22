from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import torch
from torchvision import transforms
import openai
import os
from flask_cors import CORS
from sqlalchemy import func
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True
from flask_cors import CORS
CORS(app)  # 전체 Origin 허용
CORS(app, supports_credentials=True)

# 환경 변수 또는 config 파일에서 실제로 관리하는 것이 좋음
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/woundview'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# ------------------- DB Models -------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
## class User(테이블명 users) : 사용자 정보 DB에 저장. id/username/email/created_at(가입날짜,시간)이 담겨있음

class WoundRecord(db.Model):
    __tablename__ = 'wound_records'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # user_id를 nullable로 변경
    image_path = db.Column(db.String(256))
    crop_image_path = db.Column(db.String(256))
    prediction = db.Column(db.String(128))
    advice = db.Column(db.Text)
    caution = db.Column(db.Text)
    symptom_analysis = db.Column(db.Text)
    recovery_period = db.Column(db.Text)
    care_guide = db.Column(db.Text)
    user = db.relationship('User', backref=db.backref('wound_records', lazy=True))
    wound_width = db.Column(db.Float)
    wound_height = db.Column(db.Float)
    wound_area = db.Column(db.Float)
    risk_level = db.Column(db.String(16))
    cause = db.Column(db.String(256))
    bodypart = db.Column(db.String(256))
    date = db.Column(db.DateTime, default=datetime.utcnow)
    redness = db.Column(db.Boolean)
    swelling = db.Column(db.Boolean)
    heat = db.Column(db.Boolean)
    pain = db.Column(db.Boolean)
    function_loss = db.Column(db.Boolean)

    def to_dict(self):
        """
        WoundRecord 인스턴스를 dict로 변환합니다.

        Returns:
            dict: WoundRecord의 주요 속성 딕셔너리
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'image_path': self.image_path,
            'crop_image_path': self.crop_image_path,
            'prediction': self.prediction,
            'advice': self.advice,
            'caution': self.caution,
            'symptom_analysis': self.symptom_analysis,
            'recovery_period': self.recovery_period,
            'care_guide': self.care_guide,
            'wound_width': self.wound_width,
            'wound_height': self.wound_height,
            'wound_area': self.wound_area,
            'risk_level': self.risk_level,
            'cause': self.cause,
            'bodypart': self.bodypart,
            'date': self.date.isoformat() if self.date else None,
            'redness': self.redness,
            'swelling': self.swelling,
            'heat': self.heat,
            'pain': self.pain,
            'function_loss': self.function_loss,
        }
    
## class WoundRecord(테이블명 wound_records) : 상처 정보 DB에 저장. id/user_id/image_path/date/prediction/advice/user/width/height/area/risk_level/caution이 담겨있음

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    session_id = db.Column(db.String(64), primary_key=True)
    history = db.Column(db.Text, nullable=False)  # JSON 문자열
    updated_at = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())



# ------------------- ML/LLM Functions -------------------
def predict_image(image_file):

    import os
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import timm  # EfficientNet용
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from collections import defaultdict

    class_names = ['찰과상', '멍', '화상', '찢어진 상처', '정상', '수술상처', '궤양']
    
    # 모델 로딩
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("wound_model_EN6414.0001.pth", map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    from PIL import Image
    image = Image.open(image_file).convert('RGB')
    target_img = transform(image).unsqueeze(0)

    # 예측
    with torch.no_grad():
        output = model(target_img)
        predict_result = torch.argmax(output).item()

    return {'prediction': class_names[predict_result]}

from wound_prompt import generate_advice_openai

from wound_size_yolo import estimate_wound_size


# ------------------- API Endpoints -------------------


from werkzeug.utils import secure_filename
import os

from flask_cors import cross_origin

@app.route('/upload', methods=['POST'])
def upload():
    """
    프론트엔드에서 전송된 이미지 파일을 서버에 저장하고,
    DB(wound_records)에 파일 정보와 체크리스트 데이터를 저장합니다.
    실제 예측은 /predict-image에서 처리합니다.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    save_dir = 'uploads'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    file.save(save_path)

    # 체크리스트 데이터와 answers 파싱
    checklist_data = {}
    answers = {}
    
    try:
        if 'checklist' in request.form:
            import json
            checklist_data = json.loads(request.form['checklist'])
            print(f"체크리스트 데이터: {checklist_data}")
        
        if 'answers' in request.form:
            import json
            answers = json.loads(request.form['answers'])
            print(f"증상 답변: {answers}")
    except Exception as e:
        print(f"체크리스트 데이터 파싱 오류: {e}")

    # WoundRecord 생성 시 체크리스트 데이터 포함
    new_record = WoundRecord(
        image_path=save_path,
        # 체크리스트 기본 정보
        date=datetime.fromisoformat(checklist_data.get('date').replace('Z', '+00:00')) if checklist_data.get('date') else datetime.utcnow(),
        bodypart=checklist_data.get('bodyPart'),
        cause=checklist_data.get('cause'),
        # 염증 증상 체크리스트
        redness=answers.get('redness', False),
        swelling=answers.get('swelling', False),
        heat=answers.get('heat', False),
        pain=answers.get('pain', False),
        function_loss=answers.get('function_loss', False)
    )
    db.session.add(new_record)
    db.session.commit()

    return jsonify({
        'record_id': new_record.id,
        'image_path': save_path
    }), 200

@app.route('/upload-crop', methods=['POST'])
def upload_crop():
    from werkzeug.utils import secure_filename
    import os

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    save_dir = 'uploads/crop'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    file.save(save_path)

    record_id = request.form.get('record_id')
    if record_id:
        record = WoundRecord.query.get(record_id)
        if record:
            record.crop_image_path = save_path
            db.session.commit()

    return jsonify({'crop_filepath': save_path}), 200

@app.route('/predict-image', methods=['POST'])
def predict_image_api():
    print("=== /predict-image 진입 ===")
    try:
        print("=== /predict-image try 진입 ===")
        # 1. 데이터 파싱
        data = request.get_json()
        record_id = data.get('record_id')
        original_image_path = data.get('original_image_path')
        crop_image_path = data.get('crop_image_path')

        # 2. DB에서 record 조회
        record = WoundRecord.query.get(record_id)
        if not record:
            raise Exception("해당 record_id에 대한 WoundRecord가 없습니다.")

        # 3. AI 분석 실행
        from wound_prompt import generate_advice_openai
        
        # AI 분석을 위한 기본 데이터 준비 및 DB 업데이트
        predict_result = record.prediction or "찰과상"
        if not record.prediction:
            record.prediction = predict_result
            
        # 상처 크기 기본값 설정
        if not record.wound_width:
            record.wound_width = 10  # 기본 10mm
        if not record.wound_height:
            record.wound_height = 10  # 기본 10mm
        if not record.wound_area:
            record.wound_area = record.wound_width * record.wound_height
            
        wound_size = {
            'wound_width': record.wound_width,
            'wound_height': record.wound_height,
            'wound_area': record.wound_area
        }
        
        # 사용자 입력 증상 (Boolean을 문자열로 변환)
        redness = "예" if record.redness else "아니오"
        swelling = "예" if record.swelling else "아니오"
        heat = "예" if record.heat else "아니오"
        pain = "예" if record.pain else "아니오"
        function_loss = "예" if record.function_loss else "아니오"
        
        # 기본 정보 설정
        if not record.bodypart:
            record.bodypart = "다리"
        if not record.cause:
            record.cause = "일반상처"
            
        date = record.date or "미상"
        bodypart = record.bodypart
        cause = record.cause
        
        # AI 분석 실행
        ai_result = generate_advice_openai(
            predict_result=predict_result,
            wound_size=wound_size,
            redness=redness,
            swelling=swelling,
            heat=heat,
            pain=pain,
            function_loss=function_loss,
            date=date,
            bodypart=bodypart,
            cause=cause
        )
        
        # AI 분석 결과를 record에 저장
        if 'error' not in ai_result:
            record.risk_level = ai_result.get('risk_level')
            record.symptom_analysis = ai_result.get('symptom_analysis')
            record.recovery_period = ai_result.get('recovery_period')
            record.care_guide = ai_result.get('care_guide')
            record.caution = ai_result.get('caution')
            record.advice = ai_result.get('action_guide')  # action_guide를 advice로 매핑
            
            # 기본 데이터와 AI 분석 결과 모두 DB 업데이트
            db.session.commit()
            print("=== 기본 데이터 설정 및 AI 분석 완료 ===")
        else:
            # AI 분석 실패시에도 기본 데이터는 저장
            db.session.commit()
            print(f"=== AI 분석 실패: {ai_result} ===\n=== 기본 데이터는 저장 완료 ===")

        print("=== wound_record.to_dict() 결과 ===")
        record_dict = record.to_dict()
        print(record_dict)
        
        # 이미지 경로를 절대 URL로 변환
        if record_dict.get('image_path'):
            record_dict['image_path'] = f"http://127.0.0.1:5000/{record_dict['image_path']}"
        if record_dict.get('crop_image_path'):
            record_dict['crop_image_path'] = f"http://127.0.0.1:5000/{record_dict['crop_image_path']}"
            
        print("=== 이미지 경로 변환 후 ===")
        print(f"image_path: {record_dict.get('image_path')}")
        print(f"crop_image_path: {record_dict.get('crop_image_path')}")
        print("=== /predict-image try 블록 통과 ===")
        return jsonify({'wound_record': record_dict}), 200
    except Exception as e:
        print("=== /predict-image 에러 발생 ===")
        import traceback
        traceback.print_exc()
        return jsonify({'error': '서버 내부 오류', 'detail': str(e)}), 500
    crop_image_path = data.get('crop_image_path')

    if record_id:
        record = WoundRecord.query.get(record_id)
        if not record:
            return jsonify({'error': 'Record not found'}), 404
        if not original_image_path:
            original_image_path = record.image_path

    # --- AI 분석 로직 예시 (실제 구현에 맞게 예외 처리 추가) ---
    try:
        # 실제 AI 분석 함수 호출 및 예외 처리
        # 예시: ai_result = call_ai_analysis(...)
        ai_result = None  # 실제 AI 분석 함수로 대체
        if ai_result is None:  # 분석 실패 예시
            return jsonify({'error': 'AI 서버와 통신에 실패했습니다.'}), 500

        # wound_record 업데이트 및 반환
        # record.advice = ai_result['advice'] 등 필드 업데이트
        # db.session.commit()
        print("=== /predict-image try 블록 통과 ===")
        return jsonify({'wound_record': record.to_dict()}), 200
    except Exception as e:
        print("=== /predict-image 에러 발생 ===")
        import traceback
        traceback.print_exc()
        return jsonify({'error': '서버 내부 오류', 'detail': str(e)}), 500
        if not crop_image_path:
            crop_image_path = record.crop_image_path

    # 예외처리
    if not original_image_path or not crop_image_path:
        return jsonify({'error': 'Both original and crop image paths are required'}), 400

    # 실제 예측
    wound_size = estimate_wound_size(original_image_path)
    result = predict_image(crop_image_path)

    # advice 생성 (예시: generate_advice_openai 함수 활용)
    advice = generate_advice_openai(
        result['prediction'],
        wound_size,  # dict 전체
        False, False, False, False, False,
        '', '', ''
    )
    if isinstance(advice, dict) and 'advice' in advice:
        advice = advice['advice']


    # DB 저장 전: dict 타입이면 문자열로 변환
    import json
    if isinstance(advice, dict):
        advice = json.dumps(advice, ensure_ascii=False)
    if isinstance(caution, dict):
        caution = json.dumps(caution, ensure_ascii=False)
    if isinstance(risk_level, dict):
        risk_level = json.dumps(risk_level, ensure_ascii=False)
    if isinstance(symptom_analysis, dict):
        symptom_analysis = json.dumps(symptom_analysis, ensure_ascii=False)
    if isinstance(recovery_period, dict):
        recovery_period = json.dumps(recovery_period, ensure_ascii=False)
    if isinstance(care_guide, dict):
        care_guide = json.dumps(care_guide, ensure_ascii=False)

    # DB 업데이트
    record = WoundRecord.query.get(record_id)
    if record:
        record.prediction = result['prediction']
        record.wound_width = wound_size['wound_width']
        record.wound_height = wound_size['wound_height']
        record.wound_area = wound_size['wound_area']
        record.advice = advice
        record.caution = caution
        record.risk_level = risk_level
        record.symptom_analysis = symptom_analysis
        record.recovery_period = recovery_period
        record.care_guide = care_guide
        db.session.commit()
        return jsonify({'wound_record': record.to_dict()})



@app.route('/generate-advice', methods=['POST'])
def generate_advice_api():
    data = request.get_json()
    prediction = data.get('prediction', '')
    wound_width = data.get('wound_width', '')
    wound_height = data.get('wound_height', '')
    wound_area = data.get('wound_area', '')
    redness = data.get('redness', False)
    swelling = data.get('swelling', False)
    heat = data.get('heat', False)
    pain = data.get('pain', False)
    function_loss = data.get('function_loss', False)
    date = data.get('date', '')
    bodypart = data.get('bodypart', '')
    cause = data.get('cause', '')


    advice = generate_advice_openai(
        prediction,
        wound_width,
        wound_height,
        wound_area,
        redness,
        swelling,
        heat,
        pain,
        function_loss,
        date,
        bodypart,
        cause
        )

    return jsonify({'advice': advice})


@app.route('/record', methods=['GET', 'POST'])
def record_api():
    if request.method == 'POST':
        data = request.get_json()
        user_id = data.get('user_id')
        image_path = data.get('image_path')
        prediction = data.get('prediction')
        advice = data.get('advice')
        wound_width = data.get('wound_width')
        wound_height = data.get('wound_height')
        wound_area = data.get('wound_area')
        risk_level = data.get('risk_level')
        caution = data.get('caution')
        cause = data.get('cause')
        bodypart = data.get('bodypart')
        redness = data.get('redness', False)
        swelling = data.get('swelling', False)
        heat = data.get('heat', False)
        pain = data.get('pain', False)
        function_loss = data.get('function_loss', False)
        date = data.get('date')
        wound_record = WoundRecord(
            image_path=image_path,
            prediction=prediction,
            advice=advice,
            wound_width=wound_width,
            wound_height=wound_height,
            wound_area=wound_area,
            risk_level=risk_level,
            caution=caution,
            cause=cause,
            bodypart=bodypart,
            redness=redness,
            swelling=swelling,
            heat=heat,
            pain=pain,
            function_loss=function_loss,
            date=date
        )
        db.session.add(wound_record)
        db.session.commit()
        return jsonify({'message': 'Record created'}), 201
    elif request.method == 'GET':
        record_id = request.args.get('record_id')
        if not record_id:
            return jsonify({'error': 'record_id is required'}), 400
        record = WoundRecord.query.get(record_id)
        if not record:
            return jsonify({'error': 'Record not found'}), 404
        return jsonify({'wound_record': record.to_dict()}), 200

# WoundRecord 모델에 to_dict 메서드가 없으면 아래를 클래스에 추가하세요.
# def to_dict(self):
#     return {
#         'id': self.id,
#         'user_id': self.user_id,
#         'image_path': self.image_path,
#         'prediction': self.prediction,
#         'advice': self.advice,
#         'wound_width': self.wound_width,
#         'wound_height': self.wound_height,
#         'wound_area': self.wound_area,
#         'risk_level': self.risk_level,
#         'caution': self.caution,
#         'cause': self.cause,
#         'bodypart': self.bodypart,
#         'redness': self.redness,
#         'swelling': self.swelling,
#         'heat': self.heat,
#         'pain': self.pain,
#         'function_loss': self.function_loss,
#         'date': self.date.strftime('%Y-%m-%d') if self.date else None
#     }

    return jsonify({'message': 'Record saved'})

@app.route('/chat', methods=['POST'])
def chat_api():
    """
    OpenAI 기반 상처분석 전문가 챗봇 (세션별 DB에 대화 이력 저장)
    입력: {"session_id": "...", "message": "..."}
    출력: {"response": "...", "session_id": "..."}
    """
    data = request.get_json()
    session_id = data.get("session_id")
    message = data.get("message")
    if not session_id or not message:
        return jsonify({"error": "session_id, message 모두 필요"}), 400

    # DB에서 이력 조회 또는 신규 생성
    session = ChatSession.query.get(session_id)
    if session:
        history = json.loads(session.history)
    else:
        history = [{"role": "system", "content": "너는 상처분석 전문가야"}]

    # 유저 메시지 추가
    history.append({"role": "user", "content": message})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=history,
            temperature=0.7,
            max_tokens=1024
        )
        answer = response.choices[0].message['content'].strip()
        # 챗봇 응답도 이력에 추가
        history.append({"role": "assistant", "content": answer})

        # DB에 이력 저장/업데이트
        if session:
            session.history = json.dumps(history)
        else:
            session = ChatSession(session_id=session_id, history=json.dumps(history))
            db.session.add(session)
        db.session.commit()

        return jsonify({"response": answer, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)