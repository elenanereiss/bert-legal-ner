# How to fine-tune BERT on [German LER Dataset](https://github.com/elenanereiss/Legal-Entity-Recognition)

Based on the [scripts](https://github.com/huggingface/transformers/tree/main/examples/legacy/token-classification) for token classification on the GermEval 2014 (German NER) dataset.

## Training with 19 fine-grained labels

### Preprocessing

Download the dataset splits (train, dev, test) from GitHub and save it in the `data` folder.

```bash
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_train.conll -P data
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_dev.conll -P data
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_test.conll -P data
```

Court decisions consists of long sentences that need to be edited. The `preprocess.py` script splits longer sentences into smaller ones (once the max. subtoken length is reached).

We can define some variables that we need for further pre-processing steps and training the model:

```bash
export DATA_DIR=/home/user/named-entity-recognition/data
export MODEL_DIR=/home/user/named-entity-recognition/models

export MAX_LENGTH=512
export BERT_MODEL=bert-base-german-cased

export TRAIN=$DATA_DIR/ler_train.conll
export DEV=$DATA_DIR/ler_dev.conll
export TEST=$DATA_DIR/ler_test.conll
```

Run the pre-processing script on training, dev and test datasets. Note that the script `run_ner.py` takes the following files for training and evaluation: `train.txt`, `dev.txt`, `test.txt`.

```bash
python3 preprocess.py $TRAIN $BERT_MODEL $MAX_LENGTH > $DATA_DIR/train.txt
python3 preprocess.py $DEV $BERT_MODEL $MAX_LENGTH > $DATA_DIR/dev.txt
python3 preprocess.py $TEST $BERT_MODEL $MAX_LENGTH > $DATA_DIR/test.txt
```

Then we collect all the labels from the splits. German LER has 19 fine-grained labels.

```bash
cat $DATA_DIR/ler_train.conll $DATA_DIR/ler_dev.conll $DATA_DIR/ler_test.conll | cut -d " " -f 2 | grep -v "^$"| sort | uniq >  $DATA_DIR/labels.txt
```

### Arguments

To see what arguments can be set, run `python3 run_ner.py --help`.


### Training with Pytorch

```bash
python3 run_ner.py --data_dir $DATA_DIR \
    --labels $DATA_DIR/labels.txt \
    --task_type NER \
    --model_name_or_path $BERT_MODEL \
    --output_dir $MODEL_DIR/$BERT_MODEL-$LABEL_TYPE-$MAX_LENGTH \
    --max_seq_length $MAX_LENGTH \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 12 \
    --learning_rate 1e-5 \
    --save_steps 7500 \
    --seed 1 \
    --do_train \
    --do_eval \
    --do_predict
```

Results on the dev set:

```bash
10/30/2022 17:42:03 - INFO - __main__ - ***** Eval results *****
10/30/2022 17:42:03 - INFO - __main__ -   eval_precision = 0.9212203128016991
10/30/2022 17:42:03 - INFO - __main__ -   eval_recall = 0.9458762886597938
10/30/2022 17:42:03 - INFO - __main__ -   eval_f1 = 0.9333855032769246
```

Results on the test set:

```bash
[INFO|trainer.py:2891] 2022-10-30 17:42:14,836 >> ***** Running Prediction *****
10/30/2022 17:45:18 - INFO - __main__ -   test_precision = 0.9449558173784978
10/30/2022 17:45:18 - INFO - __main__ -   test_recall = 0.9644870349492672
10/30/2022 17:45:18 - INFO - __main__ -   test_f1 = 0.9546215361725869
```


### Training with Tensorflow

```bash
python3 run_tf_ner.py --data_dir $DATA_DIR \
    --labels $DATA_DIR/labels.txt \
    --task_type NER \
    --model_name_or_path $BERT_MODEL \
    --output_dir $MODEL_DIR/$BERT_MODEL-$LABEL_TYPE-$MAX_LENGTH \
    --max_seq_length $MAX_LENGTH \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 12 \
    --learning_rate 1e-5 \
    --save_steps 7500 \
    --seed 1 \
    --do_train \
    --do_eval \
    --do_predict
```

Results on the dev set:

```
[INFO|trainer_tf.py:320] 2022-11-02 09:04:17,682 >> ***** Running Prediction *****
11/02/2022 09:05:46 - INFO - __main__ -
              precision    recall  f1-score   support

          AN       0.75      0.50      0.60        12
         EUN       0.92      0.93      0.92       116
         GRT       0.95      0.99      0.97       331
          GS       0.98      0.98      0.98      1720
         INN       0.84      0.91      0.88       199
          LD       0.95      0.95      0.95       109
         LDS       0.82      0.43      0.56        21
         LIT       0.88      0.92      0.90       231
         MRK       0.50      0.70      0.58        23
         ORG       0.64      0.71      0.67       103
         PER       0.86      0.93      0.90       186
          RR       0.97      0.98      0.97       144
          RS       0.94      0.95      0.94      1126
          ST       0.91      0.88      0.89        58
         STR       0.29      0.29      0.29         7
          UN       0.81      0.85      0.83       143
          VO       0.76      0.95      0.84        37
          VS       0.62      0.80      0.70        56
          VT       0.87      0.92      0.90       275

   micro avg       0.92      0.94      0.93      4897
   macro avg       0.80      0.82      0.80      4897
weighted avg       0.92      0.94      0.93      4897
```

Results on the test set:

```
[INFO|trainer_tf.py:320] 2022-11-02 09:11:42,672 >> ***** Running Prediction *****
11/02/2022 09:19:33 - INFO - __main__ -
              precision    recall  f1-score   support

          AN       1.00      0.89      0.94         9
         EUN       0.90      0.97      0.93       150
         GRT       0.98      0.98      0.98       321
          GS       0.98      0.99      0.98      1818
         INN       0.90      0.95      0.92       222
          LD       0.97      0.92      0.94       149
         LDS       0.91      0.45      0.61        22
         LIT       0.92      0.96      0.94       314
         MRK       0.78      0.88      0.82        32
         ORG       0.82      0.88      0.85       113
         PER       0.92      0.88      0.90       173
          RR       0.95      0.99      0.97       142
          RS       0.97      0.98      0.97      1245
          ST       0.79      0.86      0.82        64
         STR       0.75      0.80      0.77        15
          UN       0.90      0.95      0.93       108
          VO       0.80      0.83      0.81        71
          VS       0.73      0.84      0.78        64
          VT       0.93      0.97      0.95       290

   micro avg       0.94      0.96      0.95      5322
   macro avg       0.89      0.89      0.89      5322
weighted avg       0.95      0.96      0.95      5322
```

## Run as a bash script

We can type all commands in bash script and run in terminal `./run.sh`.

```bash
#!/bin/bash

export LABEL_TYPE=fine
export BERT_MODEL=bert-base-german-cased
export MAX_LENGTH=512

#can be changed if necessary
export DATA_DIR=/home/user/named-entity-recognition/data
export MODEL_DIR=/home/user/named-entity-recognition/models

# Download the splits from github
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_train.conll -P $DATA_DIR
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_dev.conll -P $DATA_DIR
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_test.conll  -P $DATA_DIR

echo "Start training with the fine-grained labels"
# Collect fine-grained labels
cat $DATA_DIR/ler_train.conll $DATA_DIR/ler_dev.conll $DATA_DIR/ler_test.conll | cut -d " " -f 2 | grep -v "^$"| sort | uniq >  $DATA_DIR/labels.txt
export TRAIN=$DATA_DIR/ler_train.conll
export DEV=$DATA_DIR/ler_dev.conll
export TEST=$DATA_DIR/ler_test.conll

# Create folder if does not exist
[[ -d $MODEL_DIR/$BERT_MODEL-$LABEL_TYPE-$MAX_LENGTH ]] || mkdir $MODEL_DIR/$BERT_MODEL-$LABEL_TYPE-$MAX_LENGTH

# Preprocess the splits
python3 preprocess.py $TRAIN $BERT_MODEL $MAX_LENGTH > $DATA_DIR/train.txt
python3 preprocess.py $DEV $BERT_MODEL $MAX_LENGTH > $DATA_DIR/dev.txt
python3 preprocess.py $TEST $BERT_MODEL $MAX_LENGTH > $DATA_DIR/test.txt

# Run training
python3 run_ner.py --data_dir $DATA_DIR \
    --labels $DATA_DIR/labels.txt \
    --task_type NER \
    --model_name_or_path $BERT_MODEL \
    --output_dir $MODEL_DIR/$BERT_MODEL-$LABEL_TYPE-$MAX_LENGTH \
    --max_seq_length $MAX_LENGTH \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 12 \
    --learning_rate 1e-5 \
    --save_steps 7500 \
    --seed 1 \
    --do_train \
    --do_eval \
    --do_predict

# Clean data folder
#rm $DATA_DIR/*
```