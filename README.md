# p3-dst-yuk
- baseline 코드
- SUMBT 추가했습니다
- wandb 연동했습니다

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

## usage
- train 할 때 architecture 변경해주면 됩니다
```python
# train
## TRADE train (batch_size 32도 가능한 것 같습니다)
python train.py --model_name_or_path "monologg/koelectra-base-v3-discriminator"

## SUMBT train (batch_size 크면 gpu 에러나는 것 같습니다)
python train.py --architecture SUMBT --model_name_or_path "dsksd/bert-ko-small-minimal"

# infernece
## model의 path만 지정해주면 될 것 같습니다
python inference.py --model_name "TRADE/model-2.pth"

```
