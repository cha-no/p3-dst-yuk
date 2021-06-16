# DST(Dialogue State Tracking)

# 목차

- [프로젝트 소개](#프로젝트-소개)
- [Model](#model-architectures)
- [회고록](#회고록)

## 프로젝트 소개


**대화 상태 추적(Dialogue State Tracking)** 은 목적 지향형 대화(Task-Oriented Dialogue)의 중요한 하위 테스크 중 하나입니다.
유저와의 대화에서 미리 시나리오에 의해 정의된 정보인 Slot과 매 턴마다 그에 속할 수 있는 Value의 집합인, 대화 상태 (Dialogue State)를 매 턴마다 추론하는 테스크입니다.
대화 상태는 아래 그림과 같이 미리 정의된 J(45)개의 Slot S마다 현재 턴까지 의도된 Value를 추론하여 (S, V)와 같은 페어의 집합(B)으로 표현될 수 있습니다. ( 이 때, 현재까지 의도되지 않은 정보(Slot)는 "none"이라는 특별한 Value를 가지게 되고, 아래 B에서 생략되어 있습니다.)

## Models

- SUMBT
- TRADE
- SOMDST
- TRADE Transformer Decoder

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

## 회고록
