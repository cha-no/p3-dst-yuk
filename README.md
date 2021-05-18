# p3-dst-yuk
### 기존
- baseline 코드
- SUMBT 추가했습니다
- wandb 연동했습니다
### 변경
- TRADE encoder부분 bert로 바꾸었습니다
- word_dropout 적용했습니다
- TRADE에서 w부분 transpose 제외했습니다
- max_seq_length 핸들링 코드 적용했습니다


## argument

### 공통 argument

hyperparameter|default| 
|:---:|:---:|
|random_seed|2020|
|data_dir|"/opt/ml/input/data/train_dataset"|
|model_dir|"models"|
|architecture|"TRADE"|
|group_decay|True|
|weight_decay|0.01|
|max_seq_length|64|
|word_dropout|0.0|
|max_label_length|12|
|train_batch_size|8|
|eval_batch_size|8|
|learning_rate|5e-5|
|adam_epsilon|1e-8|
|max_grad_norm|1.0|
|num_train_epochs|10|
|warmup_ratio|0.1|
|model_name_or_path|"monologg/koelectra-base-v3-discriminator"|

### TRADE argument

hyperparameter|default| 
|:---:|:---:|
|hidden_size|768|
|vocab_size|None|
|hidden_dropout_prob|0.1|
|proj_dim|None|
|teacher_forcing_ratio|0.5|

### SUMBT argument

hyperparameter|default|
|:---:|:---:|
|hidden_dim|300|
|num_rnn_layers|1|
|zero_init_rnn|False|
|attn_head|4|
|fix_utterance_encoder|False|
|task_name|"sumbtgru"|
|distance_metric|"euclidean"|

### SOMDST argument

hyperparameter|default|
|:---:|:---:|
|dec_learning_rate|1e-4|
|exclude_domain|True|

## usage
- train 할 때 architecture 변경해주면 됩니다
- 생각보다 메모리를 적게 잡아먹어서 batch_size를 크게 해줘도 될 것 같습니다
- batch_size를 너무 작게하면 loss가 nan이 되니깐 참고하시면 될 것 같습니다
```python
# train
## SOMDST train
train.py --architecture SOMDST --model_name_or_path dsksd/bert-ko-small-minimal --num_train_epochs 50 --train_batch_size 16 --teacher_forcing_ratio 1.0 --num_workers 2


# infernece
## model의 path만 지정해주면 될 것 같습니다
python inference.py --model_name "SOMDST/model-50.bin"

```
