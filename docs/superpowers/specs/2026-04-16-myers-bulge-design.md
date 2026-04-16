# Cas-OFFinder-rust에 Bulge 지원 추가 설계

## 배경

현재 `cas-offinder-rust`는 bit-packed popcount 기반의 brute-force 검색을 사용한다. 고정 길이 윈도우에서 pattern과 genome을 bitwise AND 후 popcount로 mismatch 수를 계산하기 때문에 **indel(bulge)을 찾을 수 없다**. CRISPR off-target 분석에서는 DNA/RNA bulge가 생물학적으로 유의미한 off-target 형태이므로 bulge 검색 기능이 필요하다.

기존 `cas-offinder-bulge`는 Python wrapper로 pattern을 변형(N 삽입)해서 원본 cas-offinder로 돌리는 방식이다. 이 접근법은 pattern 수가 수십 배 증가하여 속도가 느리다.

## 목표

1. Cas-OFFinder-rust에 **DNA bulge + RNA bulge 검색 기능** 추가
2. `cas-offinder-bulge` 대비 속도 개선
3. 결과 포맷을 `cas-offinder-bulge`와 호환되게 유지

## 채택 알고리즘

**PAM scan + Myers bit-parallel edit distance** (2-phase)

### 왜 이 조합인가

- **PAM 정확성 보장**: NGG 같은 PAM은 Cas9 결합에 필수이므로 반드시 exact match여야 한다. Bulge가 PAM 영역에 들어가면 안 된다.
- **Myers bit-parallel**: mismatch + indel을 하나의 edit distance로 계산하는 고속 알고리즘. 현재 bit-packed 인코딩과 구조적으로 잘 어울린다.
- **Cas-offinder-bulge 대비 빠름**: pattern 확장이 필요 없어 연산량이 크게 줄어든다.

### 대안 검토

| 방식 | Bulge | 속도 | 비고 |
|---|---|---|---|
| 현재 popcount | ✗ | 매우 빠름 | 기능 부족 |
| Full Smith-Waterman | ✓ | 느림 | O(m²) per site, local alignment은 CRISPR에 부적합 |
| cas-offinder-bulge | ✓ | 느림 | pattern ~56배 확장 |
| **PAM scan + Myers** | ✓ | 빠름 | 채택 |
| 통합 Myers (PAM 포함) | ✓ | 약간 더 빠름 | **PAM 영역에 bulge 허용되어 biological correctness 손상** → 기각 |

## 알고리즘 세부

### Phase 1: PAM 스캔

Genome을 stream하면서 각 position `j`에서 다음을 확인:
- `genome[j:j+pam_len]`이 PAM 패턴(예: NGG)과 exact match인지
- N은 아무 뉴클레오타이드와 매치
- 나머지 위치는 정확히 일치해야 함

구현: 4-bit 인코딩 상태에서 bitwise AND로 3개 문자 체크 (O(1))

NGG 위치는 평균 1/16 position. Human genome(3×10⁹ bp) 기준 약 400M 개의 PAM 위치.

### Phase 2: Myers bit-parallel edit distance

각 PAM 위치 `p`에 대해, guide RNA(길이 `m`)와 upstream genome 세그먼트를 Myers 알고리즘으로 비교.

**비교 대상**:
- Pattern (고정): gRNA 서열 (예: 20nt)
- Text: `genome[p - m - max_bulges : p]` (DNA bulge로 최대 `max_bulges` 만큼 길어질 수 있음)

**Myers 단계**:
1. Precompute `Peq[σ]` for each σ ∈ {A, C, G, T} (pattern의 각 문자별 bit-vector)
2. VP = all 1s, VN = 0 으로 초기화
3. Text 각 문자에 대해 bit 연산 7번으로 VP, VN 업데이트
4. 마지막 행의 HP, HN을 보고 edit distance 추적
5. `edit_distance ≤ max_edits` 인 위치를 매치로 기록

**edit distance 판정 기준**:
```
max_edits = max_mismatches + max_dna_bulges + max_rna_bulges
```

Myers 자체는 substitution/insertion/deletion을 하나의 edit로 취급하므로 DNA/RNA bulge를 구분하지 않는다. 1차로 `max_edits` 이하인 후보를 모두 찾고, 2차 traceback에서 DNA bulge / RNA bulge / mismatch를 분리한다. 만약 traceback 결과 개별 임계치(`max_mismatches`, `max_dna_bulges`, `max_rna_bulges`)를 초과하는 경우 그 매치는 드롭한다.

### Bulge 타입 구분 (Traceback)

Myers가 후보 위치를 찾은 후, **해당 위치에서만** traceback을 수행하여 alignment를 복원:
- Substitution → mismatch
- Gap in genome → RNA bulge (pattern 쪽에 extra)
- Gap in pattern → DNA bulge (genome 쪽에 extra)

Traceback은 후보 수가 적어(전체 NGG 중 극히 일부) 비용이 무시할 만하다.

## 입출력 포맷

### 입력 포맷 (cas-offinder-bulge 호환)

```
/path/to/genome
NNNNNNNNNNNNNNNNNNNNNNNGG 2 1
GGCCGACCTGTCGCTGACGC 5
CGCCAGCGTCAGCGACAGGT 5
...
```

- 1번 줄: genome 경로 (파일 또는 디렉토리, 기존과 동일)
- 2번 줄: `search_filter` + `max_dna_bulges` + `max_rna_bulges`
  - `search_filter`는 PAM 위치를 지정 (기존 cas-offinder와 동일)
  - bulge 파라미터가 없으면 0으로 기본값 (기존 cas-offinder-rust와 호환)
- 3번 줄 이후: `pattern` + `max_mismatches`

### 출력 포맷 (cas-offinder-bulge 호환)

```
#Bulge type	crRNA	DNA	Chromosome	Position	Direction	Mismatches	Bulge Size
X	GGCCGACCTGTCGCTGACGCNGG	GGCCGACCTGTCGCTGACGCAGG	chr1	1234567	+	0	0
DNA	GGCCGACCTGTCGC-TGACGCNGG	GGCCGACCTGTCGCATGACGCAGG	chr1	2345678	+	2	1
RNA	GGCCGACCTGTCGCTGACGCNGG	GGCCGACCTGTCGCGACGCAGG	chr1	3456789	-	1	1
...
```

- **Bulge type**: `X`(no bulge), `DNA`, `RNA`
- **crRNA**: gRNA 서열 (bulge가 있으면 `-`로 gap 표시)
- **DNA**: 매칭된 genome 서열 (bulge가 있으면 `-`로 gap 표시)
- **Chromosome, Position, Direction**: 기존과 동일
- **Mismatches**: substitution 수 (bulge 제외)
- **Bulge Size**: bulge 크기

### 로그 파일 (신규)

`<output_filename>.log` 파일을 **항상** 자동 생성 (output 파일과 같은 경로). 로그 파일 쓰기는 main thread 종료 직전 한 번만 수행되어 실행 시간에 영향이 거의 없다. (`<0.01s`)

```
=== Cas-OFFinder Rust (Myers bit-parallel) ===
Run date: 2026-04-16 11:45:23

Device: GPU (NVIDIA GeForce RTX 5090)
  (또는: CPU (32 threads, native Rust))

Input:
  Genome: /path/to/genome.fa
  Genome size: 3,098,825,702 bp
  Patterns: 4
  Search filter: NNNNNNNNNNNNNNNNNNNNNNNGG
  Max mismatches: 5
  Max DNA bulges: 2
  Max RNA bulges: 1

Phase 1 (PAM scan):
  NGG sites found: 387,456,123
  Elapsed: 2.341s

Phase 2 (Myers edit distance):
  Candidates checked: 387,456,123
  Matches found: 12,345
  Elapsed: 8.765s

Total elapsed: 11.107s
```

## 아키텍처

### 변경되는 파일

**cas-offinder-lib**:
- `src/search.rs` — 메인 검색 로직. popcount를 Myers로 교체
- `src/kernel.cl` — GPU 커널. Myers bit-parallel로 재작성
- `src/bit4ops.rs` — Myers용 Peq 테이블 생성 함수 추가
- 신규: `src/myers.rs` — Myers 알고리즘 구현 (CPU용)
- 신규: `src/traceback.rs` — 후보 위치에서 alignment 복원
- 신규: `src/pam_scan.rs` — Phase 1 PAM 스캔
- 신규: `src/log_writer.rs` — 로그 파일 생성

**cas-offinder-cli**:
- `src/cli_utils.rs` — 입력 파서. 2번째 줄의 bulge 파라미터 추가 파싱
- `src/main.rs` — 출력 포맷 변경 (cas-offinder-bulge 호환), 로그 파일 생성 추가

### 데이터 플로우

```
[Genome stream (chunked)]
          ↓
  [read_fasta/read_2bit]       (기존 코드 재활용)
          ↓
  [ChromChunkInfo 채널]         (기존 코드 재활용)
          ↓
  [chunks_to_searchchunk]       (기존 코드 재활용)
          ↓
  [SearchChunkInfo 채널]
          ↓
  ┌───────────────────────────────────────┐
  │  Phase 1: PAM Scan                    │
  │  - 각 position에서 search_filter 체크  │
  │  - NGG 위치 리스트 생성                │
  └───────────────────────────────────────┘
          ↓
  [PamHitList (position + direction)]
          ↓
  ┌───────────────────────────────────────┐
  │  Phase 2: Myers per PAM site          │
  │  - 각 NGG 위치에서 gRNA와 edit distance│
  │  - edit_distance ≤ threshold 수집      │
  │  - forward/reverse 양방향              │
  └───────────────────────────────────────┘
          ↓
  [MyersMatch (position, pattern_idx, edit_dist, traceback)]
          ↓
  [Traceback (bulge 타입/위치 복원)]
          ↓
  [Match 구조체 (기존 호환 + bulge 필드 추가)]
          ↓
  [출력 writer (cas-offinder-bulge 포맷)]
          ↓
  [output.txt] + [output.txt.log]
```

### GPU 커널 구조

**기존**: `find_matches` 커널이 popcount로 mismatch 계산
**변경**: `find_matches_myers` 커널이 Phase 1(PAM) + Phase 2(Myers)를 **fused로 수행**

개념상 2-phase지만 GPU 커널에서는 work-item 하나가 두 작업을 순차 처리하여 중간 버퍼를 피한다:

- 각 work-item이 하나의 genome position 처리
- work-item 내부에서:
  1. PAM (3-nt) exact match 체크 → 실패 시 early return
  2. Myers bit-parallel로 gRNA와 edit distance 계산
  3. `edit_distance ≤ max_edits` 이면 atomic_inc로 결과 기록
- 2D work-group: (genome positions) × (patterns)

Traceback은 GPU 결과 버퍼를 CPU에서 후처리하는 방식으로 구현한다 (후보 수가 적어 CPU로 충분).

**메모리**:
- `Peq[4]` (A, C, G, T별 pattern bit-vector) — constant memory
- Text window: `genome[p - m - max_bulges : p]` — local memory 또는 direct load

## 테스트 전략

### 단위 테스트

1. **Myers 알고리즘 정확성**: 다양한 edit distance 케이스에 대해 표준 DP 결과와 일치하는지 검증
   - Substitution만 (mismatch)
   - DNA bulge 1, 2
   - RNA bulge 1, 2
   - 혼합 (mismatch + bulge)
   - Edge case (pattern 전체 일치, 전체 불일치, 빈 pattern)

2. **PAM scan 정확성**: 알려진 NGG 위치 리스트와 일치하는지 검증

3. **Traceback 정확성**: Alignment 복원 결과가 실제 edit sequence와 일치하는지 검증

### 통합 테스트

1. **cas-offinder-bulge와 결과 비교**: 동일한 입력에 대해 `cas-offinder-bulge`와 우리 구현 결과가 일치해야 함 (regression test)
   - `upstream1000.fa` (작은 테스트 데이터)
   - 복수 패턴
   - bulge_dna=2, bulge_rna=1, mismatches=5 기본 설정

2. **기존 cas-offinder-rust 회귀**: bulge=0일 때 기존 결과와 동일해야 함

3. **CPU vs GPU 결과 일치**: 같은 입력에 대해 CPU/GPU 결과가 바이트 단위로 동일해야 함

### 성능 벤치마크

1. **Baseline**: 현재 popcount (bulge 없음)
2. **변경 후**: Myers (bulge=0, bulge=1, bulge=2 각각)
3. **cas-offinder-bulge**: 같은 설정에서 비교

측정 데이터:
- `upstream1000.fa` (작음, ~13KB)
- Human genome 일부 (chr22 정도, ~50MB)
- 가능하면 full human genome

## 오픈 이슈 / 위험 요소

1. **패턴 길이 제약**: Myers는 pattern이 64-bit word에 들어갈 때 가장 빠름. CRISPR gRNA는 보통 20-24nt라 문제없지만, 더 긴 pattern을 지원하려면 multi-word Myers가 필요함.

2. **메모리**: GPU에서 많은 수의 PAM 후보가 나올 수 있음 (human genome 400M개). 출력 버퍼 크기 조정 필요.

3. **Bulge가 많은 경우 false positive**: `max_bulges`가 크면 text window가 길어지고 candidate도 늘어남. 현재 `OUT_BUF_SIZE = 1<<22`로 설정되어 있는데 조정 필요할 수 있음.

4. **Reverse strand 처리**: 현재 코드는 pattern과 reverse complement pattern을 모두 검색. Myers에서도 동일하게 처리.

5. **Traceback 성능**: 후보 수가 많으면 traceback도 bottleneck이 될 수 있음. GPU에서 traceback을 구현할지, CPU에서 할지 결정 필요. (현재 계획: CPU에서 traceback. 후보 수가 많아도 한 번에 처리 가능한 범위일 것)

## 성공 기준

1. **기능**: cas-offinder-bulge와 동일한 매치 결과를 냄 (regression test 통과)
2. **속도**: `bulge_dna=2, bulge_rna=1` 설정에서 cas-offinder-bulge 대비 **5배 이상** 빠름
3. **기존 호환**: bulge=0 설정일 때 기존 cas-offinder-rust 결과와 동일
4. **CPU/GPU 일관성**: CPU와 GPU 결과가 동일

## 향후 작업

- 더 긴 pattern 지원 (multi-word Myers)
- 다른 PAM (NAG, NGAN 등) 지원
- GPU에서 traceback 병렬화
- FM-index 기반 추가 가속 (preprocessing cost 고려 필요)
