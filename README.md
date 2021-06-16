# DST(Dialogue State Tracking)

# 목차

- [프로젝트 소개](#프로젝트-소개)
- [Model](#models)
  - [SUMBT](#sumbt)
  - [TRADE](#trade)
  - [SOMDST](#somdst)
  - [TRADE Transformer Decoder](#trade-transformer-decoder)
- [Problem](#problem)
  - [회고록](#회고록)
- [Usage](#usage)

## 프로젝트 소개

### 대화 상태 추적(Dialogue State Tracking)

- 목적 지향형 대화(Task-Oriented Dialogue)의 중요한 하위 task중 하나입니다.
- 유저와의 대화에서 미리 시나리오에 의해 정의된 정보인 Slot과 매 턴마다 그에 속할 수 있는 Value의 집합인, 대화 상태 (Dialogue State)를 매 턴마다 추론하는 문제입니다.
- 대화 상태는 아래 그림과 같이 미리 정의된 J(45)개의 Slot S마다 현재 턴까지 의도된 Value를 추론하여 (S, V)와 같은 페어의 집합(B)으로 표현될 수 있습니다. 
  - ( 이 때, 현재까지 의도되지 않은 정보(Slot)는 "none"이라는 특별한 Value를 가지게 되고, 아래 B에서 생략되어 있습니다.)

<img src = "https://user-images.githubusercontent.com/59329586/122230737-dffa5880-cef4-11eb-9016-f9490d8258f1.png" width="70%" height="35%">

### 목적
- 목적은 위 그림과 같이 매 턴마다 알맞은 Dialogue State B_t를 추론해야 합니다.
- 예를들어, 위 대화에서 두번째 턴의 인풋/아웃풋은 아래와 같습니다.
  - input: ["안녕하세요.", "네. 안녕하세요. 무엇을 도와드릴까요?", "서울 중앙에 위치한 호텔을 찾고 있습니다. 외국인 친구도 함께 갈 예정이라서 원활하게 인터넷을 사용할 수 있는 곳이 었으면 좋겠어요."]
  - output: 두번째 턴의 Dialogue State
    - ["숙소-지역-서울 중앙", "숙소-인터넷 가능-yes"]

### 평가방법

- 모델은 **Joint Goal Accuracy** 와 **Slot Accuracy** , 그리고 **Slot F1 Score** 세 가지로 평가됩니다.

- Joint Goal Accuracy는 추론된 Dialogue State와 실제 Dialogue State의 set이 완벽히 일치하는지를 측정합니다. 즉, 여러 개의 Slot 중 하나라도 틀리면 0점을 받는 매우 혹독한 Metric입니다.
- 이에 반해, Slot Accuracy는 턴 레벨의 측정이 아닌 그 원소인 (Slot, Value) pair에 대한 Accuracy를 측정합니다. 심지어 아무런 Value를 추론하지 않고도 (== "none"으로 예측), 절반 이상의 점수를 낼 수 있는 매우 관대한 Metric입니다.

<img src = "https://user-images.githubusercontent.com/59329586/122238452-1aff8a80-cefb-11eb-85e3-ff2ea3a95e4a.png" width="70%" height="35%">

### 데이터셋

- 데이터셋은 WOS(Wizard Of Seoul)입니다.

## Models

### SUMBT
- Ontology Based Model으로 Dialogue State를 추론할 때 정의되어 있는 Ontology 집합에서 선택합니다.
<img src = "https://user-images.githubusercontent.com/59329586/122238940-821d3f00-cefb-11eb-879d-6912940ec73f.png" width="38%" height="38%">

### TRADE
- Open Vocab Based Model으로 Dialogue State를 추론할 때 전체 단어 집합에서 생성해내는 방식으로 추론합니다.
<img src = "https://user-images.githubusercontent.com/59329586/122239896-4b93f400-cefc-11eb-921f-6e8c5be3b31b.png" width="70%" height="35%">

### SOMDST
- Open Vocab Based Model으로 기존의 TRADE는 모든 Slot에 대해 단어(value)를 생성해냈다면 SOMDST는 Update되야할 Slot을 구해 단어(value)를 생성합니다.
<img src = "https://user-images.githubusercontent.com/59329586/122240458-c3621e80-cefc-11eb-9f7a-1bb057868a8b.png" width="70%" height="35%">

### TRADE Transformer Decoder
- 기존의 TRADE모델에서 단어를 생성해내는 layer를 GRU에서 Transformer Decoder로 바꿨습니다.
<img src = "https://user-images.githubusercontent.com/59329586/110621541-af13d480-81dd-11eb-84b4-f785af375faf.png" width="30%" height="50%">

## Problem

### 1. 과적합
SUMBT는 Ontology Based Model이기 때문에 Ontology 집합에서 value를 선택합니다. 따라서, 없는 value에 대해 많이 취약했고 실제로 문제점이 드러났습니다.

**-> Open Vocab Based Model으로 해결**

### 2. 너무 오래걸리는 학습시간
TRADE모델은 모든 Slot에 대해 value를 생성하므로 시간이 오래 걸립니다. 

**-> SOMDST로 해결**

### 3. 코드통합
SOMDST는 기존의 모델들과 데이터와 다른 WOZ데이터를 사용했기 때문에 코드 구조가 다릅니다.

**-> WOS데이터에 맞게 코드통합**

### [회고록](https://www.notion.so/d99b04e41b514b43940a9d59f79b6f58)

## usage
```python
# train
## SUBMT train
train.py --architecture SUMBT --max_seq_length 96

## TRADE train
train.py --architecture TRADE

## SOMDST train
train.py --architecture SOMDST 

## TRADE Transformer Decoder train
train.py --architecture TRADE_transformer 

# infernece
## model의 path만 지정
python inference.py --model_name "SOMDST/model-50.bin"

```
