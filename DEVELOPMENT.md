# Development Setup Guide

## MCP (Model Context Protocol) 서버 설정

이 프로젝트는 Claude Code와 함께 다음 4개의 MCP 서버를 사용합니다:

### 설치된 MCP 서버
1. **Context7** - 공식 문서 및 코드 예제 제공
2. **Playwright** - 브라우저 자동화 및 E2E 테스트
3. **Magic** - UI 컴포넌트 생성 (21st.dev)
4. **Sequential Thinking** - 체계적 사고 및 문제 해결

### 사용 방법
프로젝트 루트의 `.mcp.json` 파일에 모든 설정이 포함되어 있습니다.
새로운 환경에서는 이 파일이 자동으로 적용됩니다.

### API 키 정보
- Magic MCP의 API 키는 프로젝트에 포함되어 있습니다
- 개인 환경에서 동일한 MCP 설정을 사용할 수 있습니다

## 개발 환경 설정

### Prerequisites
- Python 3.12 (MediaPipe 호환성을 위해 필수)
- Node.js 18+
- Claude Code CLI

### 가상 환경 설정
```bash
# Python 3.12 가상환경 생성
py -3.12 -m venv venv

# 가상환경 활성화
source venv/Scripts/activate  # Windows Git Bash
# 또는
venv\Scripts\activate  # Windows CMD

# 의존성 설치
pip install -r requirements.txt
```

### 서버 실행
```bash
# 개발 서버 시작
uvicorn main:app --reload

# 디버그 모드
uvicorn main:app --reload --log-level debug
```

### 개발 명령어
- `uvicorn main:app --reload` - 개발 서버 시작
- `uvicorn main:app --host 0.0.0.0 --port 8000` - 프로덕션 서버
- `pip freeze > requirements.txt` - 의존성 업데이트

## 프로젝트 구조

```
server/
├── main.py              # FastAPI 애플리케이션 진입점
├── models.py            # Pydantic 모델
├── services.py          # 핵심 분석 로직
├── analysis.py          # 포즈 분석 알고리즘
├── pose_analyzer.py     # 포즈 감지 유틸리티
├── websocket_manager.py # WebSocket 및 Redis pub/sub
├── requirements.txt     # Python 의존성
└── venv/               # 가상 환경
```

## 기술 스택

### 핵심 기술
- **Python 3.12** - MediaPipe 호환성
- **FastAPI** - 모던 API 프레임워크
- **MediaPipe** - 포즈 감지 및 랜드마크 추출
- **OpenCV** - 비디오 처리 및 프레임 분석
- **Redis** - WebSocket pub/sub
- **uvicorn** - ASGI 서버

### 주요 의존성
- `mediapipe==0.10.14` - 포즈 감지
- `opencv-python==4.12.0.88` - 비디오 처리
- `fastapi==0.116.1` - 웹 프레임워크
- `uvicorn[standard]==0.35.0` - 서버
- `redis==6.4.0` - 캐싱 및 pub/sub