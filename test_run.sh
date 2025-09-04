DATA_NAME="EPIC"

python main.py \
    --data_name $DATA_NAME \
    --approach "HPM" \
    --batch_size 100 \
    --learning_rate 5e-5 \
    --num_epochs 20 \
    --embedding_colnames annotator \
    --sort_instances_by text_id \
    --num_fake_annotators 0 \
    --max_len 100 \
    --language_model_name roberta-base
    




