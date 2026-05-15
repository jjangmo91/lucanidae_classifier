"use client";

import { useEffect, useState } from "react";

interface SpecimenRow {
  id: string;
  upload_time: string | null;
  result_type: string;
  predicted_species: string | null;
  confidence: number | null;
  user_corrected: boolean | null;
  correct_species: string | null;
  sex: string | null;
  admin_status: string;
  is_trainable: boolean;
  image_url: string | null;
}

interface CorrectionTarget {
  id: string;
  image_url: string | null;
  predicted_species: string | null;
}

const STATUS_TABS = ["pending", "approve", "reject", "all"] as const;
type StatusTab = typeof STATUS_TABS[number];

const STATUS_STYLE: Record<string, string> = {
  pending: "text-amber-400 border-amber-500/40 bg-amber-500/10",
  approve: "text-emerald-400 border-emerald-500/40 bg-emerald-500/10",
  reject:  "text-red-400 border-red-500/40 bg-red-500/10",
  foreign: "text-sky-400 border-sky-500/30 bg-sky-500/8",
  review:  "text-violet-400 border-violet-500/30 bg-violet-500/8",
};

const SPECIES_OPTIONS = [
  { value: "Prosopocoilus_astacoides_blanchardi",  label: "두점박이사슴벌레" },
  { value: "Dorcus_titanus_castanicolor",           label: "넓적사슴벌레" },
  { value: "Lucanus_maculifemoratus_dybowskyi",     label: "사슴벌레" },
  { value: "Dorcus_hopei_binodulosus",              label: "왕사슴벌레" },
  { value: "Prosopocoilus_inclinatus_inclinatus",   label: "톱사슴벌레" },
  { value: "Dorcus_rectus_rectus",                  label: "애사슴벌레" },
  { value: "Dorcus_rubrofemoratus_rubrofemoratus",  label: "홍다리사슴벌레" },
  { value: "Prismognathus_dauricus",                label: "다우리아사슴벌레" },
  { value: "Dorcus_carinulatus_koreanus",           label: "털보왕사슴벌레" },
  { value: "Platycerus_hongwonpyoi_hongwonpyoi",    label: "원표애보라사슴벌레" },
  { value: "Dorcus_consentaneus_consentaneus",      label: "참넓적사슴벌레" },
  { value: "Aegus_laevicollis_subnitidus",          label: "꼬마넓적사슴벌레" },
  { value: "Nigidius_miwai",                        label: "뿔꼬마사슴벌레" },
  { value: "Dorcus_tenuihirsutus",                  label: "엷은털왕사슴벌레" },
  { value: "Figulus_punctatus",                     label: "길쭉꼬마사슴벌레" },
  { value: "Figulus_binodulus",                     label: "큰꼬마사슴벌레" },
];

const KO_NAMES = Object.fromEntries(SPECIES_OPTIONS.map((s) => [s.value, s.label]));

function koName(s: string | null) {
  if (!s) return "—";
  return KO_NAMES[s] ?? s.replace(/_/g, " ");
}

function fmt(iso: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

const SELECT_CLS = "w-full border border-border rounded-lg px-3 py-2 text-sm bg-background text-foreground focus:outline-none focus:border-amber-500/50";

export default function AdminPage() {
  const [tab, setTab]                     = useState<StatusTab>("pending");
  const [rows, setRows]                   = useState<SpecimenRow[]>([]);
  const [loading, setLoading]             = useState(true);
  const [acting, setActing]               = useState<string | null>(null);
  const [modalImg, setModalImg]           = useState<string | null>(null);
  const [corrTarget, setCorrTarget]       = useState<CorrectionTarget | null>(null);
  const [corrSpecies, setCorrSpecies]     = useState("");
  const [corrSex, setCorrSex]             = useState("");
  const [corrMaleForm, setCorrMaleForm]   = useState("");
  const [corrNote, setCorrNote]           = useState("");

  async function load(t: StatusTab) {
    setLoading(true);
    const res  = await fetch(`/admin/queue?status=${t}&limit=100`);
    const data = await res.json();
    setRows(data);
    setLoading(false);
  }

  useEffect(() => { load(tab); }, [tab]);

  async function act(id: string, action: string, extra: Record<string, string | null> = {}) {
    setActing(id);
    await fetch(`/admin/items/${id}/action`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ action, ...extra }),
    });
    setActing(null);
    load(tab);
  }

  function openCorrection(row: SpecimenRow) {
    setCorrTarget({ id: row.id, image_url: row.image_url, predicted_species: row.predicted_species });
    setCorrSpecies(row.predicted_species ?? "");
    setCorrSex("");
    setCorrMaleForm("");
    setCorrNote("");
  }

  async function submitCorrection() {
    if (!corrTarget || !corrSpecies) return;
    await act(corrTarget.id, "correct", {
      correct_species: corrSpecies,
      sex:             corrSex || null,
      male_form:       corrMaleForm || null,
      note:            corrNote || null,
    });
    setCorrTarget(null);
  }

  const pendingCount = tab === "pending" ? rows.length : null;

  return (
    <div className="py-8 space-y-6">

      {/* ── 이미지 확대 모달 ── */}
      {modalImg && (
        <div
          className="fixed inset-0 z-50 bg-black/85 flex items-center justify-center p-4 cursor-zoom-out"
          onClick={() => setModalImg(null)}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={modalImg} alt="specimen" className="max-w-2xl max-h-[80vh] object-contain rounded-xl shadow-2xl" />
        </div>
      )}

      {/* ── 교정 모달 ── */}
      {corrTarget && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
          <div className="bg-card border border-border rounded-2xl w-full max-w-md shadow-2xl overflow-hidden">

            {/* 헤더 */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-border/60 bg-amber-500/5">
              <p className="text-sm font-bold">관리자 교정</p>
              <button onClick={() => setCorrTarget(null)} className="text-muted-foreground hover:text-foreground text-lg leading-none">×</button>
            </div>

            <div className="p-5 space-y-4">
              {/* 이미지 미리보기 */}
              {corrTarget.image_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={corrTarget.image_url}
                  alt="specimen"
                  className="w-full max-h-52 object-contain rounded-xl border border-border bg-muted cursor-zoom-in"
                  onClick={() => setModalImg(corrTarget.image_url)}
                />
              )}

              <div className="text-xs text-muted-foreground font-mono">
                AI 예측: <span className="text-foreground font-semibold">{koName(corrTarget.predicted_species)}</span>
              </div>

              {/* 종 선택 */}
              <div className="space-y-1.5">
                <label className="text-xs text-muted-foreground font-mono uppercase tracking-wider">정확한 종 *</label>
                <select className={SELECT_CLS} value={corrSpecies} onChange={(e) => setCorrSpecies(e.target.value)}>
                  <option value="">선택하세요</option>
                  {SPECIES_OPTIONS.map((s) => (
                    <option key={s.value} value={s.value}>
                      {s.label} — {s.value.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </div>

              {/* 성별 */}
              <div className="space-y-1.5">
                <label className="text-xs text-muted-foreground font-mono uppercase tracking-wider">성별</label>
                <select className={SELECT_CLS} value={corrSex} onChange={(e) => { setCorrSex(e.target.value); setCorrMaleForm(""); }}>
                  <option value="">선택 안 함</option>
                  <option value="male">수컷</option>
                  <option value="female">암컷</option>
                  <option value="unknown">모름</option>
                </select>
              </div>

              {/* 수컷 형태 */}
              {corrSex === "male" && (
                <div className="space-y-1.5">
                  <label className="text-xs text-muted-foreground font-mono uppercase tracking-wider">수컷 형태</label>
                  <select className={SELECT_CLS} value={corrMaleForm} onChange={(e) => setCorrMaleForm(e.target.value)}>
                    <option value="">선택 안 함</option>
                    <option value="major">대형 (뿔 큰 수컷)</option>
                    <option value="minor">소형 (뿔 작은 수컷)</option>
                    <option value="intermediate">중간형</option>
                  </select>
                </div>
              )}

              {/* 메모 */}
              <div className="space-y-1.5">
                <label className="text-xs text-muted-foreground font-mono uppercase tracking-wider">메모 (선택)</label>
                <input
                  type="text"
                  className={SELECT_CLS}
                  placeholder="특이사항 입력..."
                  value={corrNote}
                  onChange={(e) => setCorrNote(e.target.value)}
                />
              </div>

              {/* 버튼 */}
              <div className="flex gap-2 pt-1">
                <button
                  onClick={submitCorrection}
                  disabled={!corrSpecies || acting === corrTarget.id}
                  className="flex-1 h-10 rounded-lg bg-amber-500 hover:bg-amber-400 text-amber-950 font-bold text-sm transition-colors disabled:opacity-40"
                >
                  교정 저장
                </button>
                <button
                  onClick={() => setCorrTarget(null)}
                  className="flex-1 h-10 rounded-lg border border-border text-sm hover:bg-muted/30 transition-colors"
                >
                  취소
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── 헤더 ── */}
      <div className="space-y-1">
        <p className="text-[10px] font-mono tracking-[0.3em] text-amber-500/45 uppercase">Admin</p>
        <h1 className="text-2xl font-black">관리자</h1>
        <p className="text-sm text-muted-foreground">업로드된 표본 검토 및 학습 데이터 관리</p>
      </div>

      {/* ── 탭 ── */}
      <div className="flex gap-1 border-b border-border/50">
        {STATUS_TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium transition-all border-b-2 -mb-px ${
              tab === t
                ? "border-amber-500 text-amber-400"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {t === "pending" ? "검토 대기" : t === "approve" ? "승인됨" : t === "reject" ? "거부됨" : "전체"}
            {t === "pending" && pendingCount !== null && pendingCount > 0 && (
              <span className="ml-1.5 text-[10px] bg-amber-500 text-amber-950 font-bold px-1.5 py-0.5 rounded-full">
                {pendingCount}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ── 테이블 ── */}
      {loading ? (
        <p className="text-sm text-muted-foreground py-8 text-center">불러오는 중...</p>
      ) : rows.length === 0 ? (
        <p className="text-sm text-muted-foreground py-8 text-center">항목 없음</p>
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground font-mono">{rows.length}건</p>
          <div className="rounded-xl border border-border overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/30">
                  <th className="px-4 py-2.5 w-16"></th>
                  {["시간", "AI 예측", "신뢰도", "유저 피드백", "상태"].map((h) => (
                    <th key={h} className="text-left px-4 py-2.5 text-xs font-mono text-muted-foreground/70 uppercase tracking-wider whitespace-nowrap">{h}</th>
                  ))}
                  {tab === "pending" && (
                    <th className="text-left px-4 py-2.5 text-xs font-mono text-muted-foreground/70 uppercase tracking-wider">액션</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={r.id} className={`border-b border-border/50 hover:bg-muted/20 transition-colors ${i % 2 ? "bg-muted/10" : ""}`}>

                    {/* 썸네일 */}
                    <td className="px-3 py-2">
                      {r.image_url ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={r.image_url}
                          alt=""
                          className="w-12 h-12 object-cover rounded-lg cursor-zoom-in hover:opacity-80 hover:scale-105 transition-all border border-border/50"
                          onClick={() => setModalImg(r.image_url)}
                        />
                      ) : (
                        <div className="w-12 h-12 rounded-lg bg-muted/50 flex items-center justify-center text-[10px] text-muted-foreground/30">없음</div>
                      )}
                    </td>

                    {/* 시간 */}
                    <td className="px-4 py-3 text-xs font-mono text-muted-foreground whitespace-nowrap">{fmt(r.upload_time)}</td>

                    {/* AI 예측 */}
                    <td className="px-4 py-3">
                      <p className="font-semibold text-sm">{koName(r.predicted_species)}</p>
                      <p className="text-[10px] italic text-muted-foreground">{r.predicted_species?.replace(/_/g, " ") ?? "—"}</p>
                    </td>

                    {/* 신뢰도 */}
                    <td className="px-4 py-3 font-mono text-amber-400 font-bold text-sm whitespace-nowrap">
                      {r.confidence !== null ? `${(r.confidence * 100).toFixed(1)}%` : "—"}
                    </td>

                    {/* 유저 피드백 */}
                    <td className="px-4 py-3">
                      {r.user_corrected === null ? (
                        <span className="text-muted-foreground/40 text-xs">없음</span>
                      ) : r.user_corrected ? (
                        <div className="space-y-0.5">
                          <span className="text-xs text-red-400 font-medium">교정: </span>
                          <span className="text-xs font-semibold">{koName(r.correct_species)}</span>
                          {r.sex && <span className="text-[10px] text-muted-foreground ml-1">({r.sex})</span>}
                        </div>
                      ) : (
                        <span className="text-xs text-emerald-400 font-medium">✓ 맞아요</span>
                      )}
                    </td>

                    {/* 상태 */}
                    <td className="px-4 py-3">
                      <span className={`text-[10px] font-medium px-2 py-0.5 rounded border ${STATUS_STYLE[r.admin_status] ?? STATUS_STYLE.pending}`}>
                        {r.admin_status}
                      </span>
                    </td>

                    {/* 액션 */}
                    {tab === "pending" && (
                      <td className="px-4 py-3">
                        <div className="flex gap-1.5 flex-wrap">
                          <button onClick={() => act(r.id, "approve")} disabled={acting === r.id}
                            className="text-[11px] px-2.5 py-1 rounded border border-emerald-700/50 text-emerald-400 hover:bg-emerald-900/20 transition-colors disabled:opacity-40">
                            승인
                          </button>
                          <button onClick={() => openCorrection(r)} disabled={acting === r.id}
                            className="text-[11px] px-2.5 py-1 rounded border border-amber-700/50 text-amber-400 hover:bg-amber-900/20 transition-colors disabled:opacity-40">
                            교정
                          </button>
                          <button onClick={() => act(r.id, "reject")} disabled={acting === r.id}
                            className="text-[11px] px-2.5 py-1 rounded border border-red-700/50 text-red-400 hover:bg-red-900/20 transition-colors disabled:opacity-40">
                            거부
                          </button>
                          <button onClick={() => act(r.id, "foreign")} disabled={acting === r.id}
                            className="text-[11px] px-2.5 py-1 rounded border border-sky-700/50 text-sky-400 hover:bg-sky-900/20 transition-colors disabled:opacity-40">
                            외래종
                          </button>
                        </div>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
