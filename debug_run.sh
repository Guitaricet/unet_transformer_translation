MODEL_NAME=transformer_en_de_toy
mkdir -p checkpoints/$MODEL_NAME

fairseq-train\
     data-bin/wmt17_en_de_toy \
     --user-dir ./unet_transformer \
     --arch unet_transformer --share-decoder-input-output-embed \
     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000\
     --dropout 0.3 --weight-decay 0.0001 \
     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
     --max-tokens 1000 \
     --max-update 100 \
     --save-dir checkpoints/$MODEL_NAME \
     --eval-bleu \
     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
     --eval-bleu-detok moses \
     --eval-bleu-remove-bpe \
     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
