# cartpole_with_DQN
DQN을 이용하여 openAI gym의 cartpole 실행하는 프로젝트입니다. <br><br>

# 프로젝트 소개

## 개발 환경 설명
Mac OS Monterey 12.4버전의 시스템에서 진행하였으며, conda 가상환경인 “cartpole”환경에서 진행되었다. 해당 환경은 export 진행하여 cartpole.yaml로 export하였다.

## DQN 설명  
### DQN의 구성
DQN이란, Q-function을 deep neural network의 parameter θ를 통해 근사화하는 것이다. 이는 optimal한 Q-value를 찾기 위해 모든 action-state pair를 찾지 않아도 된다는 장점이 있다.

Neural network를 학습시키기 위해서는 loss function이 필요한데, DQN의 loss function은 다음과 같다. <br><br>
$(R + \gamma * \max{Q(s',a';\theta)} - Q(s,a;\theta))^2$
<br><br>

<em>여기서의 θ는 prediction한 Q값(Q(s,a;θ))과 target값((R+γ*max Q(s^',a^';θ))을 같은 network에서 계산했기에 같다.</em>

DQN은 parameter θ를 경사하강법으로 갱신하면서 target값과 prediction값을 줄여나간다.

DQN은 매번 state-action pair에 대해 forward pass를 계산하는 것이 아닌,  input으로는 state를 넣고, output layer의 neuron 개수는 가능한 action의 개수만큼 설정해준다. 이때, 이동 agent의 이동 방향을 알기 위해 과거 4개의 game screen을 input으로 같이 넣어준다.

보통의 DQN에서는 state, 즉 game 화면의 각 frame image를 input으로 받기에, CNN을 사용한다. 단, object들 사이의 공간적 관계, position이 중요하기 때문에 pooling layer는 사용하지 않고 convolution layer만 사용한다.

(이번 cartpole project에서는 input(state)로 game frame image를 받지 않고, 카트의 위치,카트의 속력, 막대기의 각도, 막대기의 끝부분 속도의 값을 받기 때문에 fully-connected layer로만 구성하였다.)

### Replay Buffer
강화학습 환경에서는 state s에서 action a를 취해 state s’로 이동하고 reward r을 받는 transition이 일어나는데, DQN에서는 이 transition을 replay buffer 혹은 experience replay로 불리는 곳에 저장한다. 

이러한 replay buffer에서 randomly shuffle을 통해 골라진 batch로 학습시킨다. 그 이유는 agent의 experience는 시간 축에 따른 correlation이 강한 상태이기 때문에 randomly shuffle 없이 학습시키면 generalization이 잘 이루어지지 않아 overfitting이 일어나기 때문이다. 

이 때, experience를 sampling하는 방법에는 크게 두 가지가 있는데, 모든 experience가 같은 확률로 선택되는 uniform sampling이 있고, TD error가 높은 experience를 우선적으로 선택하는 prioritizing transition이 있다.
Replay buffer는 고정된 최근 experience들만 저장하기 때문에, queue의 형태를 가진다.

### Target network
같은 network로 target value와 predicted value를 계산하면 divergence가 일어날 수 있다. 이러한 문제를 해결하기 위해, target network라는 target value를 계산하기 위한 network를 생성한다.
Target network의 loss function은 다음과 같다. <br><br>
$(R + \gamma * \max{Q(s',a';\theta')} - Q(s,a;\theta))^2$
<br><br>
Actual network에서는 Q-value를 predict하고, θ를 경사하강법으로 수정한다. Target network는 몇 스텝동안 frozen 되어있으며, 주기적으로 actual Q-network의 weight를 복사한다. 이때, target network와 actual network는 같아야하며, target network는 actual Q를 복사한다.

### DQN의 전반적 과정 정리
1. DQN에 input으로 state를 입력함
2. epsilon-greedy policy를 이용하여 action select
3. action a를 선택한 뒤, state s에서 a를 하여 reward r을 얻고, state s’로 transition
4. 과정 3에서의 transition을 replay buffer에 <s, a, r, s’> 형태로 저장
5. replay buffer로부터 random batch를 sample하여 loss 계산
6. actual Q network의 parameter 에 대해 gradient descent를 하여 weight update
7. every k step마다 actual Q network weight 를 target network 에 복사
8. 7까지의 과정을 반복한다.

<br>

---
## Directory 설명
- main.py (DQN을 학습시켜 optimal policy를 찾는 파일)
- train.py (DQN class와 학습 function 선언된 파일)
- buffer.py (replay buffer를 구현한 파일)
- play_cartpole.py (학습된 DQN 모델로 cartpole을 플레이하는 파일)

## 실행 결과

[![DQN_cartpole_실행결과](https://img.youtube.com/vi/RBri43JX9Wc/0.jpg)](https://youtu.be/RBri43JX9Wc) 
