# 사슴벌레 AI — Frontend

Next.js 14 (App Router) 기반 프론트엔드.

## 실행

```bash
npm install
npm run dev   # http://localhost:3000
```

백엔드(`localhost:8000`)가 실행 중이어야 합니다.

## 페이지 구성

| 경로 | 설명 |
|------|------|
| `/` | 홈 — 사진 업로드 + AI 동정 |
| `/result/[id]` | 분류 결과 + 피드백 |
| `/my` | 내 도감 (localStorage 기반 이력) |
| `/species` | 종 도감 (16종 목록) |
| `/species/[slug]` | 종 상세 페이지 |
| `/admin` | 관리자 — 표본 검토·교정 |

## 환경 변수

별도 환경 변수 없음. API는 `next.config.mjs`의 rewrite로 `localhost:8000`으로 프록시됩니다.
