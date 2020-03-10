# Injecting Hierarchy with U-Net Transformers

David Donahue, Vladislav Lialin, Anna Rumshisky, 2020

Paper: [arxiv.org/abs/1910.10488](https://arxiv.org/abs/1910.10488)

This repository contains code for translation experiments.
It is a
[fairseq](https://github.com/pytorch/fairseq)
[plug-in](https://fairseq.readthedocs.io/en/latest/overview.html)
and can be easily applied to a new task.

To use it specify
`--user-dir ./unet_transformer` to extend fairseq with UNet Transformer
and
`--arch unet_transformer` to select the architecture.

To reproduce the experiments download and preprocess WMT14 EN-DE data using
[prepare-wmt14en2de.sh](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh)
(fairseq/examples/translation). And run

```bash
TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de \
    --workers 10

MODEL_NAME=transformer_en_de
mkdir -p checkpoints/$MODEL_NAME

fairseq-train\
     data-bin/wmt17_en_de \
     --user-dir ./unet_transformer \
     --arch unet_transformer --share-decoder-input-output-embed \
     --encoder-embed-dim 416 \
     --encoder-ffn-embed-dim 1664 \
     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000\
     --dropout 0.3 --weight-decay 0.0001 \
     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
     --max-tokens 3200 \
     --max-update 300000 \
     --save-dir checkpoints/$MODEL_NAME \
     --eval-bleu \
     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
     --eval-bleu-detok moses \
     --eval-bleu-remove-bpe \
     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

Training script above uses all available GPUs by default.
Parameter `--max-tokens 3200` is suited for a setup of 3x GTX 1080,
you may want to reduce this parameters and
compensate it via `--update-freq`. Number of trainable parameters: 47 531 328.

If you see `ValueError: Cannot register duplicate model (unet_transformer)`
it probably means that you already registered the model. Just remove `--user-dir`
from the train script.
