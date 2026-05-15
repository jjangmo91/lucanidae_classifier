export interface SpeciesInfo {
  slug: string;
  ko: string;
  en: string;
  scientific: string;
  rarity: 1 | 2 | 3 | 4;       // 1=흔함 2=보통 3=희귀 4=매우희귀
  sizeM: string;                 // 수컷 크기
  sizeF: string;                 // 암컷 크기
  habitat: string;               // 서식지
  season: string;                // 활동 시기
  description: string;           // 종 설명
  dimorphism: string;            // 암수 구별 포인트
  tip: string;                   // 동정 팁
  modelKey: string;              // 모델 출력 레이블
}

export const RARITY_LABEL: Record<number, string> = {
  1: "흔함",
  2: "보통",
  3: "희귀",
  4: "매우 희귀",
};

export const RARITY_COLOR: Record<number, string> = {
  1: "bg-slate-100 text-slate-700",
  2: "bg-green-100 text-green-700",
  3: "bg-amber-100 text-amber-700",
  4: "bg-red-100 text-red-700",
};

export const ALL_SPECIES: SpeciesInfo[] = [
  {
    slug: "dorcus-hopei-binodulosus",
    ko: "왕사슴벌레",
    en: "Korean stag beetle",
    scientific: "Dorcus hopei binodulosus",
    rarity: 4,
    sizeM: "40–80 mm",
    sizeF: "35–50 mm",
    habitat: "전국 산림 (주로 중부 이남 참나무림)",
    season: "6–9월",
    description: "한국에서 가장 크고 인기 있는 사슴벌레 중 하나. 광택 있는 검은 몸체와 크게 발달한 대악(큰턱)이 특징이며, 주로 참나무 수액에 모여든다.",
    dimorphism: "수컷은 대악이 크게 발달하고 앞가슴이 넓다. 암컷은 대악이 짧고 몸이 전체적으로 더 둥글다.",
    tip: "대악의 내치(안쪽 톱니)가 2개인 점이 넓적사슴벌레와의 구별 포인트.",
    modelKey: "Dorcus_hopei_binodulosus",
  },
  {
    slug: "dorcus-titanus-castanicolor",
    ko: "넓적사슴벌레",
    en: "Giant stag beetle",
    scientific: "Dorcus titanus castanicolor",
    rarity: 1,
    sizeM: "35–75 mm",
    sizeF: "35–50 mm",
    habitat: "전국 산림",
    season: "6–9월",
    description: "몸이 넓고 납작한 대형종. 왕사슴벌레와 함께 국내 최대 사슴벌레로 꼽히며, 넓은 앞가슴등판이 이름의 유래다.",
    dimorphism: "수컷 대악이 왕사슴벌레보다 내치가 적고 단순하다. 암컷은 앞가슴이 더 좁고 대악이 짧다.",
    tip: "왕사슴벌레보다 몸이 더 납작하고 넓은 느낌. 대악 내치 수로 구별.",
    modelKey: "Dorcus_titanus_castanicolor",
  },
  {
    slug: "lucanus-maculifemoratus-dybowskyi",
    ko: "사슴벌레",
    en: "Japanese stag beetle",
    scientific: "Lucanus maculifemoratus dybowskyi",
    rarity: 2,  // B
    sizeM: "40–80 mm",
    sizeF: "35–50 mm",
    habitat: "전국 산림 (참나무류 우점 지역)",
    season: "6–8월",
    description: "한국을 대표하는 사슴벌레. 붉은 다리와 길게 뻗은 대악이 특징으로, '사슴벌레'라는 이름의 주인공이다.",
    dimorphism: "수컷은 대악이 몸길이만큼 길게 발달. 암컷은 대악이 매우 짧고 다리의 붉은색이 연하다.",
    tip: "다리 넓적다리마디(퇴절)의 붉은 반점이 뚜렷하다. *Lucanus* 속 특유의 더듬이 형태 확인.",
    modelKey: "Lucanus_maculifemoratus_dybowskyi",
  },
  {
    slug: "prosopocoilus-inclinatus-inclinatus",
    ko: "톱사슴벌레",
    en: "Saw-tooth stag beetle",
    scientific: "Prosopocoilus inclinatus inclinatus",
    rarity: 2,
    sizeM: "25–65 mm",
    sizeF: "25–35 mm",
    habitat: "전국 산림",
    season: "6–9월",
    description: "대악에 톱니 모양의 돌기가 다수 발달한 것이 특징. 야행성이 강하고 수액과 발효 과일에 잘 모인다.",
    dimorphism: "수컷 대악의 톱니 수와 길이가 뚜렷하다. 암컷은 대악이 짧고 톱니가 거의 없다.",
    tip: "대악의 톱니 패턴이 가장 확실한 구별점. 몸 표면의 적갈색 광택도 특징.",
    modelKey: "Prosopocoilus_inclinatus_inclinatus",
  },
  {
    slug: "prosopocoilus-astacoides-blanchardi",
    ko: "두점박이사슴벌레",
    en: "Two-spotted stag beetle",
    scientific: "Prosopocoilus astacoides blanchardi",
    rarity: 3,  // A
    sizeM: "25–52 mm",
    sizeF: "25–35 mm",
    habitat: "남부 지방 산림 (제주도 포함)",
    season: "6–9월",
    description: "앞가슴등판 양쪽에 검은 점 2개가 있어 붙은 이름. 남부 지방에 주로 분포하며 제주도에서도 관찰된다.",
    dimorphism: "수컷 대악이 길고 안쪽으로 굽는다. 두 검은 점은 암수 모두에서 관찰된다.",
    tip: "앞가슴등판의 두 점이 가장 확실한 식별 포인트. 없거나 흐리면 다른 종 가능성.",
    modelKey: "Prosopocoilus_astacoides_blanchardi",
  },
  {
    slug: "dorcus-rectus-rectus",
    ko: "애사슴벌레",
    en: "Small stag beetle",
    scientific: "Dorcus rectus rectus",
    rarity: 1,  // C
    sizeM: "18–38 mm",
    sizeF: "18–28 mm",
    habitat: "전국 (평지~저산지)",
    season: "5–9월",
    description: "국내에서 가장 흔하게 관찰되는 소형 사슴벌레. 적응력이 뛰어나 도시 근교에서도 발견된다.",
    dimorphism: "수컷은 대악이 비교적 발달하나 다른 대형종에 비해 작다. 암수 모두 다리가 붉은색.",
    tip: "소형이고 다리가 붉은 점이 특징. 홍다리사슴벌레와 혼동되나 서식 범위가 더 넓다.",
    modelKey: "Dorcus_rectus_rectus",
  },
  {
    slug: "dorcus-rubrofemoratus-rubrofemoratus",
    ko: "홍다리사슴벌레",
    en: "Red-legged stag beetle",
    scientific: "Dorcus rubrofemoratus rubrofemoratus",
    rarity: 2,
    sizeM: "28–55 mm",
    sizeF: "25–38 mm",
    habitat: "남부 해안 및 제주도 (난대림)",
    season: "6–9월",
    description: "뚜렷한 붉은 다리가 특징인 중형 사슴벌레. 주로 남부 해안 난대림에 분포하며 분포 범위가 제한적이다.",
    dimorphism: "수컷은 대악이 발달하고 붉은 다리가 뚜렷. 암컷도 다리 색이 붉어 구별하기 쉽다.",
    tip: "퇴절(넓적다리마디) 전체가 선명한 적색. 애사슴벌레보다 크고 서식지가 남방계.",
    modelKey: "Dorcus_rubrofemoratus_rubrofemoratus",
  },
  {
    slug: "prismognathus-dauricus",
    ko: "다우리아사슴벌레",
    en: "Daurian stag beetle",
    scientific: "Prismognathus dauricus",
    rarity: 2,
    sizeM: "15–30 mm",
    sizeF: "12–22 mm",
    habitat: "전국 산지 (고지대)",
    season: "6–8월",
    description: "소형 산지성 사슴벌레. 대악이 위쪽으로 구부러진 독특한 형태를 가지며, 고지대 활엽수림에서 주로 발견된다.",
    dimorphism: "수컷 대악이 위로 크게 굽고 내치가 발달. 암컷은 대악이 짧고 굽음이 약하다.",
    tip: "*Prismognathus* 속 특유의 위로 굽은 대악 형태가 결정적 식별자.",
    modelKey: "Prismognathus_dauricus",
  },
  {
    slug: "dorcus-carinulatus-koreanus",
    ko: "털보왕사슴벌레",
    en: "Korean hairy stag beetle",
    scientific: "Dorcus carinulatus koreanus",
    rarity: 4,  // S
    sizeM: "35–65 mm",
    sizeF: "30–45 mm",
    habitat: "중부 이북 산림",
    season: "6–8월",
    description: "앞가슴등판에 황갈색 털이 밀생하는 것이 특징인 희귀종. 주로 중부 이북에 분포하며 개체 수가 많지 않다.",
    dimorphism: "수컷은 털과 대악 모두 발달. 암컷은 털이 더 짧고 대악이 단순하다.",
    tip: "앞가슴의 황갈색 털이 가장 큰 특징. 왕사슴벌레와 서식지 겹침 주의.",
    modelKey: "Dorcus_carinulatus_koreanus",
  },
  {
    slug: "platycerus-hongwonpyoi-hongwonpyoi",
    ko: "원표애보라사슴벌레",
    en: "Hongwonpyo's stag beetle",
    scientific: "Platycerus hongwonpyoi hongwonpyoi",
    rarity: 3,
    sizeM: "10–20 mm",
    sizeF: "10–18 mm",
    habitat: "산지 낙엽활엽수림",
    season: "5–7월",
    description: "국내 최소형 사슴벌레 중 하나. 보라색 금속 광택이 독특하며 한국 고유 아종이다. 한국 곤충학자 홍원표 선생의 이름을 딴 종.",
    dimorphism: "암수 모두 소형이며 보라색 광택을 띤다. 수컷의 대악이 암컷보다 약간 더 발달한다.",
    tip: "보라색~청록색 금속광택이 결정적 식별자. 소형이라 야외에서 놓치기 쉽다.",
    modelKey: "Platycerus_hongwonpyoi_hongwonpyoi",
  },
  {
    slug: "dorcus-consentaneus-consentaneus",
    ko: "참넓적사슴벌레",
    en: "True broad stag beetle",
    scientific: "Dorcus consentaneus consentaneus",
    rarity: 2,
    sizeM: "30–55 mm",
    sizeF: "25–38 mm",
    habitat: "전국 산림",
    season: "6–9월",
    description: "넓적사슴벌레와 유사하나 더 소형인 종. 전국적으로 분포하며 참나무 수액에 모여든다.",
    dimorphism: "수컷은 넓적사슴벌레보다 대악이 단순하다. 암컷은 넓적사슴벌레 암컷과 구별이 어렵다.",
    tip: "넓적사슴벌레보다 몸의 폭이 상대적으로 좁고 대악의 내치가 적다.",
    modelKey: "Dorcus_consentaneus_consentaneus",
  },
  {
    slug: "aegus-laevicollis-subnitidus",
    ko: "꼬마넓적사슴벌레",
    en: "Small flat stag beetle",
    scientific: "Aegus laevicollis subnitidus",
    rarity: 4,  // S
    sizeM: "15–28 mm",
    sizeF: "13–22 mm",
    habitat: "남부 지방 (제주도 포함)",
    season: "6–8월",
    description: "소형 납작한 사슴벌레로 남부 지방에 제한 분포한다. 썩은 나무 속에서 유충이 자라며 성충도 수피 아래에서 발견된다.",
    dimorphism: "수컷 대악이 짧고 굵다. 암컷은 더욱 납작하고 대악이 극히 짧다.",
    tip: "*Aegus* 속 특유의 납작하고 광택있는 몸체. 남방계 분포로 제주도·남해안에서 주로 발견.",
    modelKey: "Aegus_laevicollis_subnitidus",
  },
  {
    slug: "nigidius-miwai",
    ko: "뿔꼬마사슴벌레",
    en: "Miwa's stag beetle",
    scientific: "Nigidius miwai",
    rarity: 4,
    sizeM: "10–18 mm",
    sizeF: "10–16 mm",
    habitat: "제주도 및 남부 해안 일부",
    season: "5–8월",
    description: "국내 최희귀 사슴벌레 중 하나. 제주도와 남부 해안 일부에만 분포하며 생태 정보가 매우 부족하다.",
    dimorphism: "수컷은 대악이 위로 솟구치는 독특한 형태. 암컷은 대악이 매우 짧다.",
    tip: "뿔처럼 위로 솟은 대악이 가장 큰 특징. 제주도 관찰 시 식별 가능성 높음.",
    modelKey: "Nigidius_miwai",
  },
  {
    slug: "dorcus-tenuihirsutus",
    ko: "엷은털왕사슴벌레",
    en: "Thin-haired stag beetle",
    scientific: "Dorcus tenuihirsutus",
    rarity: 4,  // S
    sizeM: "35–60 mm",
    sizeF: "28–42 mm",
    habitat: "중부 이북 산림",
    season: "6–8월",
    description: "털보왕사슴벌레와 근연종이나 털이 더 성기고 짧다. 분포가 중부 이북에 한정되는 희귀종.",
    dimorphism: "수컷 대악과 털이 발달. 암컷은 털이 더 적고 대악이 짧다.",
    tip: "털보왕사슴벌레와 구별 시 털의 밀도와 길이 비교. 앞가슴등판의 세로 융기선 확인.",
    modelKey: "Dorcus_tenuihirsutus",
  },
  {
    slug: "figulus-punctatus",
    ko: "길쭉꼬마사슴벌레",
    en: "Elongated pygmy stag beetle",
    scientific: "Figulus punctatus",
    rarity: 1,  // C
    sizeM: "12–22 mm",
    sizeF: "11–18 mm",
    habitat: "남부 해안, 제주도",
    season: "5–8월",
    description: "몸이 가늘고 길쭉한 소형 사슴벌레. 남방계 분포로 주로 제주도와 남해안에서 관찰된다.",
    dimorphism: "수컷 대악이 암컷보다 약간 발달. 암수 모두 원통형 몸체.",
    tip: "길고 원통형인 몸이 *Figulus* 속의 특징. 큰꼬마사슴벌레보다 가늘다.",
    modelKey: "Figulus_punctatus",
  },
  {
    slug: "figulus-binodulus",
    ko: "큰꼬마사슴벌레",
    en: "Large pygmy stag beetle",
    scientific: "Figulus binodulus",
    rarity: 4,  // S
    sizeM: "10–20 mm",
    sizeF: "9–17 mm",
    habitat: "제주도",
    season: "5–8월",
    description: "꼬마사슴벌레 중 비교적 굵은 편. 제주도에 한정 분포하는 극희귀종으로 관찰 기록이 매우 드물다.",
    dimorphism: "수컷 대악이 암컷보다 약간 더 발달하나 차이가 작다.",
    tip: "제주도 관찰 시 *Figulus* 종 가능성 고려. 길쭉꼬마사슴벌레보다 몸이 더 굵다.",
    modelKey: "Figulus_binodulus",
  },
];

export const SPECIES_BY_SLUG = Object.fromEntries(
  ALL_SPECIES.map((s) => [s.slug, s])
);

export const SPECIES_BY_MODEL_KEY = Object.fromEntries(
  ALL_SPECIES.map((s) => [s.modelKey, s])
);
