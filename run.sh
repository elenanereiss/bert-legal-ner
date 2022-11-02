#!/bin/bash

# Arguments from terminal
export LABEL_TYPE=$1
export BERT_MODEL=$2

export MAX_LENGTH=512

#can be changed if necessary
export MODEL_DIR=/netscratch/leitner/named_entity_recognition/models
export DATA_DIR=/home/leitner/ner/data

# Download the splits from github
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_train.conll -P $DATA_DIR
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_dev.conll -P $DATA_DIR
wget https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/ler_test.conll  -P $DATA_DIR

if [[ "$LABEL_TYPE"=="fine" ]]
then
  echo "Start training with the fine-grained labels"
  # Collect fine-grained labels
  cat $DATA_DIR/ler_train.conll $DATA_DIR/ler_dev.conll $DATA_DIR/ler_test.conll | cut -d " " -f 2 | grep -v "^$"| sort | uniq >  $DATA_DIR/labels.txt
  export TRAIN=$DATA_DIR/ler_train.conll
  export DEV=$DATA_DIR/ler_dev.conll
  export TEST=$DATA_DIR/ler_test.conll
elif [[ "$LABEL_TYPE"=="coarse" ]]
then
  echo "Start training with the coarse-grained labels"
  # Generate coarse-grained labels
  python3 generate_coarse_labels.py $DATA_DIR/ler_train.conll $DATA_DIR/lerc_train.conll
  python3 generate_coarse_labels.py $DATA_DIR/ler_dev.conll $DATA_DIR/lerc_dev.conll
  python3 generate_coarse_labels.py $DATA_DIR/ler_test.conll $DATA_DIR/lerc_test.conll
  
  # Collect coarse-grained labels
  cat $DATA_DIR/lerc_train.conll $DATA_DIR/lerc_dev.conll $DATA_DIR/lerc_test.conll | cut -d " " -f 2 | grep -v "^$"| sort | uniq >  $DATA_DIR/labels.txt
  export LABELS=$DATA_DIR/labels.txt
  export TRAIN=$DATA_DIR/lerc_train.conll
  export DEV=$DATA_DIR/lerc_dev.conll
  export TEST=$DATA_DIR/lerc_test.conll
else
  echo "ERROR: label types are fine or coarse"
  exit
fi

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
    --overwrite_output_dir \
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
