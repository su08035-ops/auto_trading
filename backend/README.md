# auto_trading

일봉 기반 단일 종목 DQN / Double DQN 연구용 실험 프로그램이다.
YAML 설정 -> 데이터 -> 피처 -> 학습 -> 검증으로 모델 선택 -> 테스트와 기준전략 비교 -> 결과 저장까지 연결되어 있다.

## 1. 처음 한 번 실행

Mac 터미널을 열고 프로젝트 폴더로 이동한다.

```bash
cd /Users/hawnwoongku/Desktop/auto_trading
python -B main.py --demo --episodes 3
```

이 명령은 API 키 없이 명시적으로 만든 합성 OHLCV로 3에피소드를 실행한다.
삼성전자 실제 주가나 수익성 검증 결과가 아니다. 한국 거래소 휴일도 반영하지 않은
평일 데이터이며, 프로그램 연결 확인용이다. API 오류 때 자동으로 합성 데이터로
바꾸지 않는다. `--demo`를 쓰거나 `data.source: synthetic`으로 명시해야 한다.

패키지가 없다면 프로젝트 폴더의 같은 터미널에서 설치한다.

```bash
python -m pip install -r requirements.txt
```

## 2. 실제 한투 데이터로 실행

`config/config.yaml`의 `data.source` 기본값은 `kis`다.
이전에 키를 export한 바로 그 터미널에서 다음을 실행한다.

```bash
python -B main.py --episodes 3
```

해당 터미널에 키가 아직 없다면 먼저 아래 환경변수를 설정한다.
키는 Python 파일이나 config.yaml에 넣지 않는다.

```bash
export KIS_APP_KEY="발급받은_APP_KEY"
export KIS_APP_SECRET="발급받은_APP_SECRET"
python -B main.py --episodes 3
```

`kis_api.environment`와 키의 실전/모의 구분을 일치시킨다.
시세 조회용 실전 키를 사용해도 실험에서 실제 주문은 전송하지 않는다.
키가 없거나 API가 실패하면 이유를 표시하고 종료한다.
다른 터미널이나 Codex 실행 환경은 이 터미널의 export를 자동으로 공유하지 않는다.

OHLCV CSV를 가지고 있다면 경로를 지정한다.

```bash
python -B main.py --csv /absolute/path/ohlcv.csv --episodes 3
```

CSV에는 `date, open, high, low, close, volume` 컬럼이 필요하다.
`date`는 `2018-01-02`처럼 날짜로 해석할 수 있어야 한다.
실행할 때마다 내려받은/사용한 OHLCV 사본이 결과 폴더에 저장된다.

## 3. 현재 기본 실험 설계

| 구분 | 설정 |
|---|---|
| 종목 | 삼성전자 005930, KIS 또는 사용자가 지정한 CSV |
| 데이터 요청 | 2018-01-01 ~ 2025-12-31 |
| 학습 | 2023-01-01 이전, 지표와 윈도우 준비 이후 |
| 검증 | 2023-01-01 이상 ~ 2024-01-01 미만 |
| 테스트 | 2024-01-01 이상 ~ 2025-12-31 |
| 상태 | 15개 시장 피처 x 20일 + 보유비중 = 301 |
| 행동 | 목표 주식 비중 0%, 50%, 100% |
| 보상 | 비용 반영 총자산의 로그수익률 |
| 학습 시작 | 처음 경험 256개 수집 후 |
| 배치 | 학습 한 번에 경험 64개 |
| 반복 | 기본 100에피소드, CLI `--episodes`로 이번 실행만 변경 |
| 모델 선택 | 검증 누적수익률이 가장 높은 에피소드, 동률은 먼저 나온 모델 |

데이터 끝을 2025-12-31로 고정하고, 기존 train/test 경계 2024-01-01 앞의
2023년을 검증용으로 분리했다. 이 날짜들은 config에서 변경할 수 있다.
실제 첫 체결일/마지막 체결일과 스텝 수는 각 실행의 manifest.json에 기록된다.

검증/테스트 시작 직전 20개 피처 행은 초기 상태를 만드는 문맥으로만 사용한다.
검증/테스트의 주문과 보상은 해당 구간 첫 거래일 시가부터 시작하므로
평가를 위한 윈도우 준비 때문에 첫 20일을 누락하지 않는다.
검증과 테스트 데이터는 리플레이 버퍼에 저장하지 않는다.
테스트 결과를 보고 이번 실행의 모델을 선택하지 않는다.

## 4. config 값은 어디로 전달되는가

```text
main.py: load_config()로 YAML 읽기
    -> 이번 실행의 CLI 변경값 적용
    -> train.py: build_agent()
    -> DQNAgent 또는 DoubleDQNAgent.from_config()
    -> __init__에 설정값을 명시적으로 전달
```

```yaml
agent:
  algorithm: dqn
  hidden_dim: 128
  optimizer: adam
  loss: mse
```

`optimizer`는 `adam` 또는 `sgd`, `loss`는 `mse`, `huber`, `mae`를 지원한다.
`hidden_dim`은 두 은닉층의 공통 크기다. 실제 실험 경로에서 이 세 설정이
빠지거나 지원하지 않는 이름이면 오류를 낸다. 생성자의 기본값으로 조용히 대체하지 않는다.
직접 객체를 만드는 단위 테스트에는 생성자 기본값을 사용할 수 있다.

`training.train_frequency: 1`은 환경 한 스텝마다 학습 요청,
`gradient_steps: 1`은 요청당 최대 한 번 Policy 업데이트라는 뜻이다.
Target 500회와 epsilon 감소는 성공한 Policy 업데이트를 기준으로 센다.
에피소드가 바뀌어도 Agent, 버퍼, epsilon과 카운터를 유지한다.
`data.synthetic_seed`는 합성 시장을 만드는 시드이고, `training.seed`와 독립적이다.
따라서 학습 시드만 바꾸어도 비교할 합성 시장은 그대로다.

## 5. DQN과 Double DQN 실행

```bash
python -B main.py --demo --episodes 3 --algorithm dqn
python -B main.py --demo --episodes 3 --algorithm double_dqn
```

실제 주가를 사용하려면 `--demo`를 뺀다. `--episodes`를 생략하면 YAML의 100을 사용한다.
`--seed 7`처럼 학습 시드를 바꿀 수 있다. CLI로 바꾼 설정은 원본 config.yaml을
수정하지 않고 해당 실행 폴더의 config.yaml에 저장한다.
이 두 번의 짧은 실행만으로 알고리즘 우열이나 논문 가설을 확정할 수는 없다.

## 6. 결과 확인

실행 완료 시 출력되는 새 결과 폴더 안에 다음 항목이 생긴다.

```text
results/<실행시각>_<알고리즘>_<데이터출처>/
  config.yaml                 실제 사용한 설정
  manifest.json               날짜, 시드 관련 설정 참조, 데이터/코드 해시, 패키지 버전, 상태
  ohlcv.csv                   해당 실행에 사용한 데이터 사본
  training_history.csv        에피소드별 손실, 검증 결과, epsilon, 업데이트 횟수
  models/dqn_model.pth        검증에서 선택한 모델 (설정된 파일명)
  models/last_model.pth       마지막 에피소드 모델
  test/metrics.json          모든 지표와 부가 계좌 통계
  test/metrics.csv           config의 evaluation.metrics로 선택한 지표
  test/equity.csv            전략별 자산 곡선
  test/agent_trades.csv       에이전트의 매 스텝 체결/보유 기록
  test/cash_trades.csv
  test/buy_and_hold_trades.csv
  results.png                자산, 낙폭, 학습 손실 그래프
  summary.md                 이번 실행 결과 요약
```

결과 폴더는 실행마다 새로 만들어서 이전 실험을 덮어쓰지 않는다.
`--output-dir /absolute/path/new-run`으로 직접 지정할 수도 있지만, 이미 존재하면 오류다.
모델 파일은 가중치와 설정을 재사용하는 평가용 체크포인트다.
리플레이 버퍼와 optimizer 전체 상태를 저장하지 않으므로 정확한 학습 재개 기능은 아니다.

저장한 모델을 다시 평가하거나 과거 데이터 모의운용을 재생할 수 있다.

```bash
python -B main.py --demo --mode evaluate --checkpoint /absolute/path/models/dqn_model.pth
python -B main.py --demo --mode paper --checkpoint /absolute/path/models/dqn_model.pth
```

커스텀 날짜로 학습했다면 `--config /absolute/path/run/config.yaml`도 함께 지정한다.
이때 구조/피처/행동/비용은 모델과 함께 저장한 설정을 읽어서 동일하게 적용한다.
평가 구간을 저장된 테스트 시작일보다 앞으로 당길 수는 없다.
`paper`는 저장된 정책으로 과거 데이터를 시간순으로 재생하는 로컬 모의운용이며,
실시간 시세를 계속 감시하는 서비스나 한투 서버의 모의주문을 뜻하지 않는다.

## 7. 지표 정의

- 누적수익률: 마지막 종가 평가자산 / 초기자산 - 1.
- MDD: 이전 최고 자산 대비 최대 하락폭의 양수 크기. 그래프는 음수 낙폭을 표시한다.
- Sharpe: 단순 일간 초과수익률 평균 / 표본 표준편차 x sqrt(252).
- Sortino: 단순 일간 초과수익률 평균 / 하방편차 x sqrt(252). 하방편차는 전체 관측일에 대해 음수 초과수익률의 제곱 평균을 사용한다.
- 거래 횟수: 실제 주식 수가 바뀐 체결 횟수. 단순 보유는 제외한다.
- 회전율: 각 거래의 거래대금 / 거래 직전 자산을 누적한다.
- 평균 보유일: FIFO로 대응한 매도 수량의 보유 거래일을 수량 가중 평균한다. 아직 팔지 않은 주식 수는 별도 기록한다.

일간 빈도 252와 무위험수익률 0은 evaluation 설정이다.
변동성/하방편차가 0이거나 매도한 수량이 없으면 해당 비율/보유일은 JSON null이다.
이 값을 성과 0으로 바꾸지 않는다.
Buy & Hold는 첫 시가에 한 번 매수한 수량을 유지하며, 매일 100%로 재조정하지 않는다.
모든 전략은 마지막 날 강제 매도 없이 종가 평가액으로 비교한다.

## 8. 파일별 책임

| 파일 | 실행에서 맡는 일 |
|---|---|
| main.py | YAML, 실행 옵션, 데이터, 실험 폴더, 전체 순서 |
| data/loader.py | KIS/CSV/명시적 합성 데이터와 YAML 읽기 |
| data/preprocessing.py | OHLCV 검사, 날짜 분할 및 평가 초기 문맥 |
| data/features.py | 시장 지표 계산. RSI의 상승/하락/횡보 및 준비 구간 처리 |
| env/trading_env.py | 거래일 진행, 301차원 상태, 계좌 평가와 reward |
| broker/paper_broker.py | 환경과 백테스트가 공유하는 현금/주식/수수료 계산 |
| agents/network.py | 상태에서 행동별 Q값으로 순전파 |
| agents/dqn.py | 경험, epsilon, MSE/Huber/MAE, optimizer, Target 갱신 |
| agents/double_dqn.py | Policy가 선택하고 Target이 평가하는 TD 목표 |
| training/train.py | 에피소드 반복, 검증 모델 선택과 체크포인트 |
| training/evaluate.py | 학습 없는 평가, 결과 파일 및 그래프 저장 |
| backtest/backtester.py | 동일한 날짜와 체결 조건으로 정책/기준전략 실행 |
| backtest/metrics.py | 수익률, 위험, 거래 및 보유일 통계 |
| broker/real_broker.py | KIS 잔고 조회와 명시적 지정가 주문 API 어댑터 |

## 9. KIS 주문 어댑터의 범위

한투 공식 [현금주문 예제](https://github.com/koreainvestment/open-trading-api/blob/main/examples_llm/domestic_stock/order_cash/order_cash.py)와
[잔고조회 예제](https://github.com/koreainvestment/open-trading-api/blob/main/examples_llm/domestic_stock/inquire_balance/inquire_balance.py)의 요청 형식을 사용한다.
`KISBroker.from_config(config, mode="paper")`는 KIS 모의 서버,
`mode="real"`은 실전 서버를 선택한다. 각각 해당 서버용 키가 필요하다.

계좌 번호와 상품 코드는 `KIS_ACCOUNT_NO`(8자리), `KIS_ACCOUNT_PRODUCT_CODE`(2자리)
환경변수에서 읽는다. 초기 설정의 `broker.allow_orders: false`에서는 주문 함수가
호출되어도 HTTP 주문 요청을 보내지 않는다. 실험 CLI는 이 어댑터를 호출하지 않는다.

어댑터는 잔고 연속조회와 명시적 지정가 주문 접수를 구현했다.
응답의 accepted는 접수이지 체결 완료가 아니다. 타임아웃 시 자동 재주문하지 않는다.
중복 client_order_id 차단은 현재 프로세스 안에서만 적용한다.
실시간 시세 루프, 취소/정정, 미체결/부분체결 조정, 재시작 후 주문 복구와
손실 제한을 포함한 실거래 운영 시스템은 이번 과거 데이터 실험 프로그램의 범위 밖이다.
실제 서버 주문은 실행하지 않았고 테스트에서는 HTTP 응답을 모의했다.

## 10. 테스트와 연구 메모

```bash
python -B -m unittest discover -s tests -v
```

설정 전달, DDQN 목표, 256개 시작, 500회 Target 갱신, 비용/보유일 기록,
평가 날짜 경계, 검증 모델 선택, 모델 재로딩, FIFO 보유일, 기준전략,
모의 HTTP 요청과 전체 데모 실행을 검사한다.

현재 설정은 최소보유제약 없이 포지션 감축을 허용한다. 체결은 다음 시가에 즉시 된다고 가정하며
슬리피지/호가/거래정지/부분체결은 모델링하지 않았다. 비용은 config의 연구 가정이다.
수익성 결론 전에는 실제 데이터, 여러 학습 시드와 미사용 검증 절차가 필요하다.
