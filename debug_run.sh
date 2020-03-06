MODEL_NAME=transformer_en_de_toy
mkdir -p checkpoints/$MODEL_NAME

fairseq-train\
     data-bin/wmt17_en_de_toy \
     --user-dir ./unet_transformer \
     --arch unet_transformer --share-decoder-input-output-embed \
     --encoder-embed-dim 16 \
     --encoder-ffn-embed-dim 32 \
     --encoder-attention-heads 4 \
     --decoder-layers 2 \
     --decoder-attention-heads 4 \
     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000\
     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
     --max-tokens 1000 \
     --max-update 100 \
     --save-dir checkpoints/$MODEL_NAME \
     --eval-bleu \
     --eval-bleu-args '{"beam": 2, "max_len_a": 1.2, "max_len_b": 10}' \
     --eval-bleu-detok moses \
     --eval-bleu-remove-bpe \
     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
