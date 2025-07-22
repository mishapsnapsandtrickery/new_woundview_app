# WoundView App 🏥

AI 기반 상처 분석 및 관리 애플리케이션

## 📋 프로젝트 구조

```
f_new_woundview_app/
├── woundview-backend/     # FastAPI 백엔드 서버
├── woundview_pj/         # React Native 프론트엔드
└── README.md            # 이 파일
```

## 🚀 빠른 시작

### 1. 필수 모델 파일 다운로드

**중요:** AI 분석을 위해 SAM(Segment Anything Model) 파일이 필요합니다.

#### SAM 모델 다운로드:
1. **[SAM 모델 다운로드 (Google Drive)](https://drive.google.com/file/d/1mSDSxE5y0lsdnPO9bkr-TDOl8XQ7R0g_/view?usp=drive_link)**
2. 다운로드한 `sam_vit_l_0b3195.pth` 파일을 `woundview-backend/` 폴더에 저장

```bash
# 올바른 위치 확인
woundview-backend/
├── sam_vit_l_0b3195.pth  ← 여기에 저장
├── app.py
├── wound_prompt.py
└── ...
```

### 2. 백엔드 설정

```bash
cd woundview-backend

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python app.py
```

### 3. 프론트엔드 설정

```bash
cd woundview_pj

# 의존성 설치
npm install

# React Native 실행
npm start
```

## 🔧 주요 기능

- **AI 상처 분석**: SAM 모델을 활용한 정확한 상처 영역 분석
- **상처 측정**: 상처 크기, 면적 자동 계산
- **진단 및 조언**: AI 기반 상처 단계 분류 및 관리 조언
- **기록 관리**: 상처 치료 과정 추적 및 기록

## 📱 API 구조

자세한 API 문서는 `woundview_pj/README.md`를 참조하세요.

## ⚠️ 주의사항

1. **Node.js 버전**: 최소 v23 이상 필요
2. **네이티브 의존성**: [React Native 환경 설정](https://reactnative.dev/docs/environment-setup#installing-dependencies) 필수
3. **SAM 모델**: 1.19GB 크기의 대용량 파일이므로 별도 다운로드 필요

## 🛠️ 개발 환경

- **백엔드**: Python, FastAPI, PyTorch
- **프론트엔드**: React Native, TypeScript
- **AI 모델**: SAM (Segment Anything Model)

## 📞 문의

프로젝트 관련 문의사항이 있으시면 Issues를 통해 연락해주세요.
