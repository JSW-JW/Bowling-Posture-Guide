# E2E Tests for Bowling Posture Guide

이 디렉토리는 Playwright를 사용한 End-to-End 테스트를 포함합니다.

## 📋 테스트 개요

### 테스트 대상 기능 (2025-08-11~13 커밋 기반)

1. **볼링 분석 기능** (`bowling-analysis.spec.js`)
   - 메인 UI 요소 표시
   - 비디오 파일 업로드 인터페이스
   - 키보드 단축키 가이드
   - 사용자명 및 룸 컨트롤

2. **룸 관리 기능** (`room-management.spec.js`)
   - 룸 선택 모달 열기/닫기
   - 새 룸 생성
   - 기존 룸 목록 표시
   - 폼 유효성 검사
   - 에러 처리

3. **채팅 시스템** (`chat-system.spec.js`)
   - 채팅 토글 기능
   - 룸별 격리된 채팅
   - 실시간 메시지 전송
   - WebSocket 연결 관리
   - 사용자 목록 표시

4. **통합 시나리오** (`integration.spec.js`)
   - 전체 워크플로우 테스트
   - 다중 사용자 시나리오
   - 페이지 새로고침 시 상태 유지
   - 에러 상태 처리
   - 빠른 사용자 상호작용

## 🏗️ 테스트 아키텍처

### 파일 구조
```
tests/
├── constants.js          # UI 텍스트, 셀렉터, 테스트 데이터 상수
├── helpers.js            # 공통 테스트 헬퍼 함수들
├── bowling-analysis.spec.js
├── room-management.spec.js  
├── chat-system.spec.js
├── integration.spec.js
├── fixtures/             # 테스트용 파일들
└── README.md            # 이 파일
```

### 설계 원칙

1. **변수화된 상수**: UI 텍스트와 셀렉터를 `constants.js`에 중앙화
2. **재사용 가능한 헬퍼**: 공통 로직을 `TestHelpers` 클래스에 추상화
3. **유지보수성**: 텍스트 변경 시 테스트 실패 방지
4. **가독성**: 테스트 코드에서 비즈니스 로직이 명확히 드러남
5. **구체적 셀렉터**: 스코프가 명확한 CSS 셀렉터 사용으로 요소 충돌 방지
6. **순차적 상호작용**: UI 상태 변화에 따른 적절한 대기 및 검증

## 🚀 실행 방법

### 사전 요구사항
```bash
# 서버 실행 (포트 8000)
cd ../server
uvicorn main:app --reload

# 클라이언트 실행 (포트 3000) - 새 터미널에서
cd client
npm start
```

### 테스트 실행
```bash
# 모든 테스트 실행
npm run test:e2e

# UI 모드로 실행 (테스트 선택 및 시각적 실행)
npm run test:e2e:ui

# 헤드 모드로 실행 (브라우저 창 표시)
npm run test:e2e:headed

# 디버그 모드로 실행
npm run test:e2e:debug

# 특정 테스트 파일만 실행
npx playwright test bowling-analysis.spec.js

# 특정 브라우저로만 실행
npx playwright test --project=chromium
```

## 📝 테스트 작성 가이드

### 1. 상수 사용하기

```javascript
// ❌ 나쁜 예 - 하드코딩된 텍스트
await expect(page.locator('h1')).toHaveText('🎳 볼링 자세 분석 AI');

// ✅ 좋은 예 - 상수 사용
await expect(page.locator(SELECTORS.MAIN_TITLE)).toHaveText(UI_TEXT.MAIN_TITLE);
```

### 2. 헬퍼 함수 활용하기

```javascript
// ❌ 나쁜 예 - 반복적인 코드
await page.locator('.room-btn').click();
await page.locator('input[placeholder="Room 이름"]').fill('테스트 룸');
await page.locator('button[type="submit"]').click();

// ✅ 좋은 예 - 헬퍼 사용
await helpers.createRoom('테스트 룸');
```

### 3. 비동기 대기 처리

```javascript
// WebSocket 연결 대기
await helpers.waitForWebSocketConnection();

// 룸 참여 대기  
await helpers.waitForRoomJoin();

// 메시지 표시 대기
await helpers.waitForMessage('테스트 메시지');
```

## 🔧 설정 파일

### `playwright.config.js`
- 브라우저 설정 (Chrome, Firefox, Safari)
- 개발 서버 자동 실행
- 베이스 URL 및 타임아웃 설정
- 리포터 설정

### 주요 설정
```javascript
// 자동 서버 실행
webServer: [
  { command: 'npm start', port: 3000 },
  { command: 'cd ../server && uvicorn main:app --host 127.0.0.1 --port 8000', port: 8000 }
]
```

## 🐛 문제 해결

### 일반적인 이슈

1. **서버가 실행되지 않음**
   ```bash
   # 서버 수동 실행 확인
   cd ../server
   uvicorn main:app --reload
   ```

2. **WebSocket 연결 실패**
   - Redis 서버 실행 상태 확인
   - 방화벽 설정 확인
   - useWebSocket cleanup 이슈: 컴포넌트 언마운트 시 적절한 정리 확인

3. **테스트 타임아웃**
   - `waitForTimeout` 값 조정
   - 네트워크 상태 확인
   - 로딩 상태 테스트 시 네트워크 지연 모킹 사용

4. **셀렉터 변경으로 인한 실패**
   - `constants.js`에서 `SELECTORS` 업데이트
   - 실제 DOM 구조와 비교
   - 범용 셀렉터 대신 구체적인 스코프 셀렉터 사용 (예: `.create-room-form button[type="submit"]`)

5. **모달 관련 테스트 실패**
   - 백드롭 클릭 기능 사용 가능 (2025-08-15 추가)
   - 폼이 표시된 후 상호작용하도록 순서 확인

### 디버깅 팁

```bash
# 스크린샷과 함께 실패한 테스트 확인
npx playwright show-report

# 특정 테스트만 디버그 모드로 실행
npx playwright test --debug -g "should create room"

# 헤드리스 모드 비활성화
npx playwright test --headed --slowMo=1000
```

## 📊 테스트 커버리지

### 현재 커버리지
- ✅ UI 요소 표시 및 상호작용
- ✅ 룸 생성 및 관리 (백드롭 클릭 포함)
- ✅ 실시간 채팅 기능
- ✅ WebSocket 연결 및 정리 로직
- ✅ 로딩 상태 및 에러 처리
- ✅ 반응형 디자인
- ✅ 네트워크 지연 모킹을 통한 로딩 상태 테스트
- ⏳ 실제 비디오 업로드 및 분석 (테스트 비디오 파일 필요)
- ⏳ 서버 API 응답 시간 테스트

### 향후 개선 사항
1. 실제 볼링 비디오 파일을 이용한 E2E 테스트
2. API 모킹을 통한 더 안정적인 테스트
3. 성능 테스트 추가
4. 접근성 테스트 추가
5. 다국어 지원 테스트 (필요시)

### 최근 개선사항 (2025-08-15)
- ✅ WebSocket cleanup 메서드 수정으로 연결 안정성 향상
- ✅ 백드롭 클릭으로 모달 닫기 기능 추가
- ✅ 테스트 셀렉터 구체화로 요소 선택 정확도 향상
- ✅ 네트워크 지연 모킹으로 로딩 상태 테스트 안정화

## 🔗 관련 문서
- [Playwright 공식 문서](https://playwright.dev/)
- [프로젝트 README](../README.md)
- [서버 API 문서](../../server/README.md)