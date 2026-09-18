# 실습 1: Q-network 직접 만들기

현재는 사용자의 요청에 따라 network.py와 DQN 구현을 완성했다.
아래 TODO 1/2는 최초 실습 항목의 이름이며, 현재 코드에는 미구현 TODO가 없다.
완성 코드의 함수별 해설은 [핵심 모듈 해설](02_core_modules.md)을 참고한다.

## 이번 실습의 목표

`agents/network.py`에 `QNetwork`를 구현한다. 환경에서 받은 숫자를 입력하고,
각 행동의 예상 가치를 나타내는 Q값을 반환하는 신경망이다.

```text
시장 feature 15개 x 최근 20일 + 현재 보유비중 1개
                         |
                      입력 301개
                         |
              Linear -> ReLU: 128개
                         |
              Linear -> ReLU: 128개
                         |
                    Linear: 3개
                         |
              Q(주식 0%), Q(50%), Q(100%)
```

128은 첫 실습에서 사용할 은닉층 크기이며, 최적값으로 검증된 수치는 아니다.
현재의 301과 3은 프로젝트 설정에서 결정되는 크기다. 클래스 안에 고정하지 않고
생성자의 `state_dim`, `action_dim`으로 받는다.

## 언제, 어디서, 어떤 방식으로?

- 언제: DQN 알고리즘을 구현하기 전에 진행한다.
- 어디서: 편집기에서 `agents/network.py`를 연다. 터미널은 프로젝트 폴더에서 실행한다.
- 어떤 방식으로: 완성된 생성자와 forward를 읽으며 아래 TODO 1/2의 요구사항과 대응시킨다.
- 이번에 수정할 파일: `agents/network.py` 한 개다. 테스트는 통과 기준으로 사용한다.

## 입출력 규격

```python
network = QNetwork(state_dim=301, action_dim=3, hidden_dim=128)
q_values = network(state)
```

| 입력 | 출력 | 의미 |
|---|---|---|
| `float32` Tensor `(301,)` | Tensor `(3,)` | state 하나에 대한 Q값 3개 |
| `float32` Tensor `(64, 301)` | Tensor `(64, 3)` | 서로 다른 state 64개를 한 번에 계산 |
| Tensor `(2, 7)`, 생성자 크기 `7, 4, 32` | Tensor `(2, 4)` | 다른 feature 수와 행동 수에도 대응 |

출력은 확률이 아니므로 음수이거나 1보다 클 수 있다. 가장 큰 Q값의 행동을
선택하는 일과 epsilon-greedy 탐험은 다음 실습인 `agents/dqn.py`에서 맡는다.

## 직접 구현할 두 부분

### TODO 1: 생성자에서 계층 등록

1. 제공된 `super().__init__()`를 유지한다.
2. `state_dim -> hidden_dim -> hidden_dim -> action_dim`의 Linear 계층 3개를 만든다.
3. 첫 번째와 두 번째 Linear 뒤에 ReLU를 배치한다.
4. 계층을 `self`의 속성으로 등록한다. `nn.Sequential`로 묶거나 각각 이름을 붙여도 된다.
5. 출력 Linear 뒤에는 활성화 함수를 붙이지 않는다. Linear의 bias는 기본값을 사용한다.

힌트: `nn.Linear(입력_크기, 출력_크기)`, `nn.ReLU()`, `nn.Sequential(...)`을 찾아본다.
완성 코드를 복사하기 전에 각 계층의 입력과 출력 크기를 종이에 적어 본다.

### TODO 2: forward에서 계산

1. 전달받은 `state`를 등록한 계층에 순서대로 통과시킨다.
2. 마지막 Linear의 결과를 Tensor 그대로 반환한다.
3. `argmax`, `softmax`, 마지막 ReLU를 추가하지 않는다.
4. 결과에 `.detach()`, `.numpy()`, `.item()`을 사용하지 않는다. 학습 때 기울기가 필요하다.
5. `forward()` 안에서 계층을 새로 만들지 않는다. 생성자에서 만든 계층을 사용한다.

PyTorch Linear는 입력의 마지막 차원에 적용되므로 단일 state와 batch를 위해
별도의 반복문을 만들 필요가 없다. `network(state)`가 `forward()`를 호출한다.

## 실행해서 확인하기

터미널에서 아래 두 줄을 순서대로 실행한다. 실제 주식 데이터나 API 키는 필요 없다.

```bash
cd /Users/hawnwoongku/Desktop/auto_trading
python -B -m unittest discover -s tests -p test_network.py -v
```

현재 완성된 코드로 실행하면 `Ran 8 tests`와 `OK`가 나와야 한다.
아래 표는 기존 신경망 단독 테스트 8개의 의미를 설명한다.

| 테스트 | 확인하는 것 |
|---|---|
| 단일 state | 숫자 301개를 받아 유한한 Q값 3개 반환 |
| batch | 64개 state를 각각 계산하고 `(64, 3)` 유지 |
| 크기 변경 | 생성자 인자로 입력·은닉층·출력 크기 변경 |
| 은닉층 계산 | 두 은닉층에서 ReLU가 실제로 적용됨 |
| Q값의 범위 | 음수와 1보다 큰 값을 출력할 수 있음 |
| 학습 가능성 | 손실에서 기울기가 흐르고 가중치가 변경됨 |
| 저장·복원 | 저장한 가중치를 불러오면 같은 결과가 나옴 |
| 환경 연결 | 현재 TradingEnv의 초기·다음 state를 바로 입력할 수 있음 |

테스트 통과는 신경망의 연결과 학습 가능성을 확인하며 수익성 검증은 아니다.
batch 학습 테스트의 목표값은 검사 전용이다. 실제 DQN의 TD 목표는 현재
agents/dqn.py에 구현되어 있으며 tests/test_dqn.py에서 별도로 검증한다.

## 막혔을 때

- `TODO 1` 오류: 생성자의 계층 정의를 완성하고 그 아래의 예외를 제거한다.
- `TODO 2` 오류: `forward()`에서 계층을 호출하고 결과를 반환한다.
- 행렬 크기 오류: 앞 계층의 출력 크기와 뒤 계층의 입력 크기를 비교한다.
- 출력값 범위 테스트 실패: 마지막에 활성화 함수나 행동 선택을 붙였는지 확인한다.
- 기울기 테스트 실패: 계층이 `self`에 등록됐는지, 계산 도중 Tensor를 분리했는지 확인한다.

완료 후에는 입력 shape가 왜 `(301,)`와 `(64, 301)` 두 가지인지, 출력이 왜
확률이 아닌지, 계층을 왜 생성자에 만드는지를 자신의 말로 설명해 본다.

## 다음 실습

현재 환경, DQN, Double DQN, 학습 루프, 평가/백테스트와 main 연결까지 완성했다.
실험 실행 방법은 README에, TD, Monte Carlo, n-step 방식의 차이는
핵심 모듈 해설에 정리했다.

참고: [PyTorch 공식 DQN 튜토리얼의 Q-network](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#q-network)
