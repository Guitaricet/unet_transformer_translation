MODEL_NAME=unet_transformer_en_de_debug_02Oct20
rm -rf checkpoints/$MODEL_NAME
mkdir -p checkpoints/$MODEL_NAME

fairseq-train\
     data-bin/wmt17_en_de_toy \
     --user-dir ./unet_transformer \
     --arch unet_transformer --share-decoder-input-output-embed \
     --encoder-layers 4 \
     --encoder-embed-dim 128 \
     --encoder-ffn-embed-dim 512 \
     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 1 \
     --dropout 0.1 --weight-decay 0.0001 \
     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
     --max-tokens 3200 \
     --max-update 20 \
     --save-dir checkpoints/$MODEL_NAME \
     --wandb-project unet_transformer_debug \
     --log-interval 1 \

#     --ddp-backend=no_c10d \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
