# 환경, 신경망, DQN: 구현 코드 해설

## 0. 이번에 완성한 범위

- `env/trading_env.py`: 일봉 데이터로 계좌와 체결을 시뮬레이션한다.
- `agents/network.py`: 상태를 받아 행동별 Q값을 계산한다.
- `agents/dqn.py`: 행동 선택, 경험 저장, TD 학습, Target 복사를 수행한다.
- `tests/`: 실제 API 없이 계산과 세 모듈의 연결을 검증한다.

이 문서는 핵심 세 모듈을 설명한다. 현재는 train.py/main.py, Double DQN,
평가와 결과 저장까지 연결되어 있다. 최신 실행 방법과 브로커 범위는
[프로젝트 실행 가이드](../README.md)를 참고한다.

아래에서 `self`는 '이 객체가 기억하는 값', `__init__`은 '처음 만들 때의 준비',
`@classmethod from_config`는 '설정 사전으로 객체를 만드는 진입점'이라고 읽으면 된다.
`@property`는 함수 계산 결과를 `env.state_dim`처럼 속성으로 읽게 해준다.
이름 앞 `_`는 다른 파일에서 직접 호출하기보다 내부에서 사용하는 보조 함수라는 관례다.

## 1. 세 파일의 연결

```text
config.yaml + 전처리된 feature_df
              |
       TradingEnv.from_config
              |
     env.state_dim = 301, env.action_size = 3
              |
       DQNAgent.from_config
              |
       QNetwork 객체 두 개 생성
       Policy: 학습 대상 / Target: 목표 계산용 복사본

환경 state
   -> agent.select_action(state)
   -> env.step(action)
   -> agent.remember(state, action, reward, next_state, done)
   -> agent.learn()
   -> 다음 state에서 반복
```

환경은 DQN을 import하지 않는다. 신경망은 환경을 import하지 않는다.
DQN만 QNetwork를 import해서 사용한다. 호출하는 쪽에서 상태와 행동을 전달해
서로 연결한다. 나중에 그 반복을 담당할 파일이 `training/train.py`다.

## 2. env/trading_env.py: 매매 결과를 계산하는 환경

### 2-1. 입력과 보관하는 값

생성자는 지표 계산이 끝난 DataFrame을 받는다. 날짜는 인덱스이며,
최소한 `open`, `close`, 설정된 시장 피처 컬럼이 필요하다.
환경 안에서 API를 호출하거나 MA/RSI를 새로 계산하지 않는다.
120일 이평 계산에 필요한 준비 데이터와 결측치 정리는 앞 단계에서 처리한다.

| 속성 | 의미 |
|---|---|
| `feature_values` | 시장 피처만 추출한 float32 배열 |
| `open_prices`, `prices` | 각각 시가, 종가 배열 |
| `current_step` | 현재 관측할 수 있는 마지막 거래일의 행 번호, 0부터 시작 |
| `cash`, `shares` | 현금과 보유 주식 수 |
| `asset_value` | 현재 종가 기준 현금 + 주식 평가액 |
| `holding_days` | 마지막 추가 매수 이후 완료한 보유 거래일 수 |
| `trade_history` | 날짜, 요청 행동, 실제 체결, 비용과 제약 적용 기록 |
| `portfolio_values` | 초기 자산 및 매 스텝 종료 자산 |

`TradeExecution`은 한 번의 체결 결과를 묶는 dataclass다. 계산 주체가 아니라
체결 방향, 주식 수, 수수료, 세금, 회전율 등을 이름 붙여 전달하는 기록이다.

### 2-2. reset(): 에피소드 시작

초기 자금 1,000만 원, 보유 주식 0주, 거래 기록 없음으로 초기화한다.
20일을 첫 상태로 묶어야 하므로 현재 행은 `window - 1`, 즉 19부터 시작한다.
상태를 반환할 뿐, 신경망을 만들거나 초기화하지 않는다.

같은 DQNAgent 객체를 에피소드 밖에서 한 번 만들고 유지하면,
환경을 reset해도 가중치, 버퍼, epsilon, 학습 횟수는 그대로 남는다.

### 2-3. _get_observation(): 301개 숫자 만들기

```text
현재가 t일 종가 이후라면

시장: t-19일부터 t일까지 20행 x 15개 피처
   -> 날짜순, 각 날짜 안에서는 설정의 피처순으로 펼침
   -> 300개 숫자

계좌: t일 종가 기준 보유주식 평가액 / 총자산
   -> 1개 숫자

합치면 shape (301,)
```

`reshape(-1)`은 행렬을 한 줄로 펼치고, `np.concatenate`는 시장 정보와
계좌 정보를 이어 붙인다. float32는 신경망 입력과 자료형을 맞추기 위한 것이다.
미래의 시가/종가나 다음 행의 피처는 현재 상태에 포함하지 않는다.

### 2-4. step(action): 환경 한 스텝

`action`은 비중 자체가 아닌 정수 인덱스다.

| action | 목표 주식 비중 |
|---|---|
| 0 | 0% |
| 1 | 50% |
| 2 | 100% |

이번 구현에서 시간 기준을 명확히 정리했다.

```text
t일 종가까지의 상태로 행동 선택
       |
t+1일 시가에서 목표 비중으로 체결
       |
t+1일 종가로 총자산 평가
       |
reward = log(t+1일 종가 자산 / t일 종가 자산)
       |
t+1일까지의 상태, reward, done, info 반환
```

이전 환경은 시장 피처를 한 행 늦게 보여주면서, 계좌 보유비중에는 그보다
나중의 가격을 사용했다. 지금은 시장 정보와 계좌 정보의 관측 시점을 맞췄다.
당일 종가를 보고 같은 종가에 체결했다고 가정하는 문제도 피하도록 다음 시가를 사용한다.

예: 초기 자산 1,000원, 다음 시가 200원, 다음 종가 220원, 비용 0일 때
100% 행동을 선택하면 5주를 산다. 다음 종가 자산은 1,100원이고 보상은
`log(1100 / 1000)`이다. 이 숫자는 단순한 단위 테스트 예시다.

기존 보유 주식은 밤사이 종가에서 시가로 바뀐 가격의 영향도 받는다.
그래서 보상의 분모는 체결 직전 시가 자산이 아니라 이전 종가 자산이다.
거래비용은 이미 cash에서 차감되므로 reward에서 또 차감하지 않는다.

### 2-5. _rebalance_to_target(): 실제 계좌 변경

1. 시가 기준 총자산과 목표 비중으로 목표 주식 수를 계산한다.
2. 목표 주식 수와 현재 주식 수의 차이를 구한다.
3. 줄이는 주문이면 현재 설정에서는 즉시 처리한다.
4. 늘리는 주문이면 수수료까지 포함해 현금으로 살 수 있는 수량을 확인한다.
5. 현금, 주식 수, 수수료, 매도세, 거래 기록을 갱신한다.

1주 단위로만 거래하며 빚을 내거나 공매도하지 않는다. 목표 비중은 비용 차감 전
총자산을 기준으로 정하므로 정수 수량과 비용 때문에 실제 비중은 정확히 50%나
100%가 아닐 수 있다. `requested_target`과 `executed_target`을 함께 기록한다.
`turnover`는 이번 거래 금액 / 체결 직전 총자산이며, `total_turnover`에 누적한다.

### 2-6. 보유일 기록의 의미

현재 설정은 `min_holding_days: 0`이므로 최소보유제약을 적용하지 않는다.
목표 비중이 낮아지면 다음 시가에 바로 포지션을 줄일 수 있다.
매수 당일 종가에 보유일은 1이 되며, 이 값은 성과 분석의 평균 보유일 계산에 사용한다.
추가 매수하면 전체 포지션의 보유일 카운터가 다시 시작한다.

보유일 수는 state에 넣지 않았다. 현재는 이 값이 체결 가능 여부를 바꾸지 않으므로
관측 상태와 주문 가능성이 보유일 제약 때문에 어긋나는 문제는 없다.

### 2-7. 종료와 입력 검증

마지막 행까지 도달하면 `done=True`다. 종료 후 step을 다시 부르면 오류를 내며
reset이 필요하다. 마지막 자산은 종가 평가액이고, 종료만을 이유로 강제 매도하지 않는다.
현재 환경은 데이터 끝을 유한 에피소드의 종료로 정의한다. 임의로 잘라낸
학습 구간을 단순 시간 제한으로 다루려면 나중에 terminated/truncated를 구분해야 한다.

정리된 피처 데이터가 N행이고 윈도우가 W라면 `episode_steps = N - W`다.
최소 `W + 1`행이 있어야 한 번 진행할 수 있다.

`_validate_inputs`는 중복 날짜, 없는 컬럼, NaN/무한대, 0 이하 가격,
잘못된 비용이나 행동 설정을 먼저 거절한다. 이는 학습 도중 NaN이 퍼지는 것을 막는다.

## 3. agents/network.py: Q값을 계산하는 신경망

### 3-1. __init__(): 계층을 한 번 만든다

```text
state_dim -> hidden1 -> ReLU -> hidden2 -> ReLU -> output -> action_dim
   301         128                128                3
```

`nn.Linear`가 학습 가능한 가중치와 편향을 가진다. 입력을 가중치로 조합하고
편향을 더한다. ReLU는 음수를 0으로 바꾸는 비선형 함수다. ReLU가 없다면
여러 Linear를 겹쳐도 전체적으로는 하나의 선형 변환과 같아진다.
출력층은 활성화 함수를 붙이지 않는다. Q값은 확률이 아니라 할인된 미래 보상의
기댓값 추정이므로 음수와 1보다 큰 값도 허용해야 한다.

`hidden_dim=128`은 첫 구현의 기본값이며 최적값이라는 뜻은 아니다.
현재 `agent.hidden_dim` 설정을 from_config가 필수로 읽는다.
optimizer와 loss도 YAML에서 각각 adam/sgd, mse/huber/mae로 선택한다.

### 3-2. forward(): 현재 가중치로 계산한다

`network(state)`를 호출하면 PyTorch가 `forward(state)`를 실행한다.
이 함수는 가중치를 새로 만들거나 업데이트하지 않는다.

| 입력 | 출력 |
|---|---|
| 상태 하나 `(301,)` | 행동 3개의 Q값 `(3,)` |
| 상태 64개 `(64, 301)` | 경험별 행동 Q값 `(64, 3)` |

`forward` 안에서 argmax나 detach를 하지 않는다. 행동을 고르는 일은 DQN이 맡고,
학습할 때에는 출력부터 각 계층까지 기울기를 계산할 경로가 필요하기 때문이다.
같은 가중치로 같은 입력을 넣으면 이 신경망은 같은 Q값을 반환한다.

### 3-3. Policy와 Target은 클래스가 아니라 객체 두 개

둘 다 QNetwork 구조를 사용한다. DQNAgent가 Policy를 만든 뒤 deepcopy해서
별개의 Target을 만든다. 시작할 때 숫자는 같지만 메모리 저장 공간은 다르다.
Policy가 학습해도 Target은 자동으로 바뀌지 않는다.

## 4. agents/dqn.py: 행동과 학습을 관리하는 알고리즘

### 4-1. __init__() / from_config(): 준비

준비하는 것은 Policy, Target, Adam optimizer, 설정된 loss, replay buffer,
epsilon 및 두 카운터다. 작은 MLP의 학습 흐름을 이해하기 쉽도록 CPU를 사용한다.
랜덤 시드가 같으면 초기 가중치와 탐험/샘플링을 재현할 수 있도록 구성했다.

| 값 | 출처 및 갱신 기준 |
|---|---|
| `gamma=0.99` | 설정: TD 목표의 미래 가치 할인 |
| `learning_rate=0.001` | 설정: Adam의 학습률 |
| `batch_size=64` | 설정: learn 한 번에 뽑는 경험 수 |
| `replay_buffer_size=20000` | 설정: 보관 개수 상한 |
| `learning_starts=256` | 설정: 처음 학습 전 누적 수집 경험 수 |
| `target_update_interval=500` | 설정: 성공한 Policy 업데이트 횟수 간격 |
| `epsilon` | 시작 1.0, 성공한 Policy 학습마다 0.999배, 최소 0.05 |
| `total_steps` | remember 한 번이 정상 완료될 때 1 증가 |
| `learning_steps` | optimizer.step 한 번이 정상 완료될 때 1 증가 |

`episodes`는 DQN 내부에서 사용하지 않는다. 에피소드 반복은 이후 train.py가 맡는다.
현재 기본 설정은 MSE와 Adam이며, YAML의 loss와 optimizer로 변경할 수 있다.
아래 학습식 설명은 기본 MSE를 기준으로 한다.

### 4-2. select_action(): 행동 선택만

학습용 호출은 `select_action(state)`다. 처음 256개 경험을 모을 때에는 무작위 행동으로
수집한다. 이후에는 epsilon 확률로 무작위, 나머지 확률로 Policy argmax를 사용한다.
무작위 분기에서는 굳이 신경망 Q값을 계산하지 않는다.

평가용 호출은 `select_action(state, explore=False)`다. 처음부터 greedy로 선택하며,
epsilon이나 카운터를 변경하지 않는다. 평가는 remember/learn도 호출하지 않아야 한다.

### 4-3. remember(): 경험의 사실만 저장

`Experience`는 필드 이름이 붙은 튜플이다. `(s, a, r, s', done)`을 저장한다.
`a`는 요청한 행동 인덱스다. 환경의 제약으로 실제 체결이 막힌 경우에도
그 요청을 했을 때의 결과를 학습해야 하므로 실제 비중으로 덮어쓰지 않는다.
행동 당시의 Q값은 저장하지 않는다. 나중에 현재 Policy로 계산한다.

버퍼는 `deque(maxlen=20000)`이므로 가득 차면 가장 오래된 경험부터 버린다.
상태 배열은 복사해서 저장한다. 호출한 쪽에서 원본 배열을 수정해도 과거 경험이
바뀌지 않게 하기 위해서다. 누적 수집 카운터는 버퍼가 가득 차도 계속 증가한다.

### 4-4. learn(): 네 노트의 7단계

처음에는 `total_steps < 256`이므로 None을 반환하고 아무것도 학습하지 않는다.
학습 시작 조건을 agent 내부에도 둬서, 이후 train.py가 매 스텝 learn을 호출해도
준비 전에 학습하는 실수를 막는다. 버퍼에 batch_size만큼 있는지도 확인한다.
별도의 스케줄러는 없으며, learn을 한 번 부르면 최대 한 번 업데이트한다.

**7-1: 64개 경험을 무작위로 뽑고 Tensor로 변환**

| 이름 | shape | 내용 |
|---|---|---|
| states | `(64, 301)` | 경험별 당시 상태 |
| actions | `(64,)` | 경험별 당시 요청 행동 인덱스 |
| rewards | `(64,)` | 실제 환경 보상 |
| next_states | `(64, 301)` | 경험별 다음 상태 |
| dones | `(64,)` | 경험별 종료 여부 |

한 배치 내에서는 중복 없이 뽑는다. 다음 학습 배치에서 같은 경험이 다시 뽑힐 수 있다.

**7-2: 현재 Policy로 기록된 행동 평가**

`all_q_values = policy_network(states)`의 결과는 `(64, 3)`이다.
`actions.unsqueeze(1)`은 행동 인덱스를 `(64, 1)`로 바꾼다.
`gather(1, ...)`는 각 행에서 그 행의 당시 행동에 해당하는 열을 선택한다.
`squeeze(1)`로 다시 `(64,)`로 만든 것이 `predicted_q`다.
배치 크기가 1이어도 배치 축이 사라지지 않도록 squeeze에 축을 지정했다.

**7-3: _calculate_td_targets()에서 목표 계산**

```text
종료가 아니면: target_q[i] = r[i] + gamma * max Q_target(s'[i], a')
종료이면:      target_q[i] = r[i]
```

미래 Q값의 max는 행동 축에서 경험마다 따로 구한다. 경험들 사이에서 최대값을
고르는 것이 아니다. 종료된 행에는 Target을 호출할 필요가 없다.
`torch.no_grad()`는 Target 목표에 기울기 경로를 만들지 않게 한다.
Target 파라미터의 requires_grad도 False로 고정했다.

**7-4: TD 오차**

`td_errors = target_q - predicted_q`로 경험별 오차 64개를 계산한다.
현재 예측이 목표보다 작으면 양수, 크면 음수다.

**7-5: Loss**

`loss_function(predicted_q, target_q)`는 개별 오차의 제곱을 평균낸다.
mean_td_error는 설명/진단용 결과일 뿐, 이 평균을 제곱해서 손실로 사용하지 않는다.
예를 들어 오차 +2와 -2는 평균 0이지만 MSE는 4다.

**7-6: 역전파 후 가중치 수정**

```text
loss 계산
  -> optimizer.zero_grad(): 이전 gradient 지우기
  -> loss.backward(): 이번 loss의 Policy gradient 계산
  -> optimizer.step(): Adam이 실제 가중치 수정
  -> learning_steps += 1
```

Adam은 단순히 학습률 x gradient만 빼는 것보다 복잡하며, 이전 기울기의 통계도 사용한다.
Loss와 gradient가 유한한지 검사하여 NaN/무한대 상태에서 업데이트가 진행되지 않게 했다.

### 4-5. Target 복사와 epsilon 감소

Policy 학습을 마친 뒤 누적 learning_steps가 500의 배수이면
`update_target_network()`가 최신 Policy state_dict를 Target에 복사한다.
500번째 업데이트의 결과를 복사하는 것이며, 500번째 업데이트 전에 복사하지 않는다.
500은 환경 스텝이나 에피소드 수가 아니다.

epsilon은 성공한 학습마다 감소한다. 경험만 수집하는 준비 기간이나
평가 호출에서는 감소하지 않는다. 이 감소 단위도 실험 설정의 일부이므로 고정해서 기록한다.

### 4-6. TD 외의 방식과 Double DQN

전에 DQN을 만들 때 다시 설명하기로 했던 차이는 다음과 같다.

- 현재: one-step TD. 실제 보상 한 개와 다음 상태의 추정값으로 바로 목표를 만든다.
- Monte Carlo: 종료까지 얻은 실제 보상들의 할인합으로 목표를 만든다. 끝까지 기다려야 하고, 일반 DQN의 one-step 목표와 다르다.
- n-step TD: 실제 보상을 n개 합친 뒤 n스텝 뒤의 Q값 추정을 붙인다.
- Double DQN: 다음 행동의 선택은 Policy argmax, 그 행동의 평가는 Target으로 분리한다. 여전히 TD 방식이다.

현재 일반 DQN과 Double DQN을 모두 구현했다. train.py의 build_agent가
config의 algorithm에 맞는 클래스를 고르고 각 클래스의 from_config를 호출한다.
서로 잘못된 클래스에 설정을 전달하면 오류를 낸다.

## 5. 두 에피소드의 연결 예시

아래는 학습 루프의 핵심 연결 예시다. `config`는 YAML을 읽은 사전,
`feature_df`는 실제 피처 생성과 날짜 분할을 마친 학습 전용 데이터라고 가정한다.
현재 train.py/main.py가 이 순서와 검증·결과 저장을 실행한다.

```python
from env.trading_env import TradingEnv
from agents.dqn import DQNAgent

env = TradingEnv.from_config(feature_df, config)
agent = DQNAgent.from_config(env.state_dim, env.action_size, config)

for episode in range(config["training"]["episodes"]):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        learning_info = agent.learn()
        state = next_state
```

Agent 생성이 for문 밖에 있다는 점이 중요하다. 안에서 새로 만들면 에피소드마다
가중치, 버퍼, epsilon, 카운터가 초기화되어 지금 논의한 학습과 달라진다.

한 에피소드가 1,000스텝이라면 첫 에피소드에서 745번 학습하고,
두 번째에서는 1,000번 학습한다. 첫 에피소드의 Target 복사는 Policy 500회째에,
두 번째는 누적 1,000회와 1,500회째에 이루어진다.

## 6. 언제, 어디서, 무엇을 실행하나

코드를 수정한 뒤 Mac 터미널에서 다음 테스트를 실행하면 된다.
실제 주가 다운로드, API 키, 증권 계좌가 필요 없다.

```bash
cd /Users/hawnwoongku/Desktop/auto_trading
python -B -m unittest discover -s tests -v
```

기존 신경망 8개, DQN 14개, 환경 11개 외에 파이프라인과 브로커 테스트를 추가했다.
합성 OHLCV -> 기존 피처 생성 -> 301차원 환경 -> 3개 에피소드 DQN 학습까지 포함한다.
이 테스트의 통과는 계산과 연결의 검증이며 수익성 검증은 아니다.

읽는 순서는 `network.py` 전체, `env`의 reset/step, `dqn`의 select_action/remember/learn,
마지막으로 검증 함수와 보조 함수 순서가 좋다. 네 노트의 7-1부터 7-6 번호를
dqn.py 주석에도 맞춰 두었다.

## 7. 연구 및 실거래 전에 남은 사항

- 전체 실행, 기록과 모델 저장은 연결했다. 실제 데이터로 연구 결과를 검증해야 한다.
- validation 기간을 설정에 추가했다. 여러 시드의 비교와 하이퍼파라미터 연구는 별도 수행해야 한다.
- 같은 날짜를 여러 에피소드 반복해도 새로운 시장 데이터가 늘어나는 것은 아니다.
- 거래비용은 설정값을 사용하는 연구 가정이며, 세율의 현재 정확성이나 종목별 예외를 이번에 확인한 것은 아니다.
- 시가에 목표 비중을 즉시 체결할 수 있다고 가정한다. 실제 주문 지연, 슬리피지, 호가, 거래정지, 미체결, 부분체결은 구현하지 않았다.
- 마지막 자산은 미청산 평가액이다. 반드시 청산한 수익률이 필요하면 별도 종료 규칙과 비용을 추가해야 한다.
- 신경망에 별도 정규화기를 추가하지 않았다. 기존 비율 피처를 사용하며, 스케일 조정이 필요하면 학습 구간에서만 적합해야 한다.
- 데이터 끝의 종료 정의를 논문에서 밝혀야 한다.
- KIS 잔고/지정가 주문 어댑터를 추가했지만 실제 주문 API는 호출하지 않았다. 운영 범위는 README에 명시했다.

기본 원리 참고: [PyTorch 공식 DQN 튜토리얼](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
튜토리얼은 Huber loss와 soft Target 업데이트를 사용하지만, 이 프로젝트의 첫 버전은
설명한 학습식을 따르기 위해 MSE와 500회 간격 hard copy를 사용한다.
