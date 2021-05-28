# p3-dst-yuk

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

## Experimental records are summarized in Wiki
