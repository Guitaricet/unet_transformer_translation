import time
import torch
import numpy as np

from fairseq import options, tasks
from unet_transformer import UNetTransformerEncoder


def timeitnow(fn, n_repeats=10):
    times = []
    for _ in range(n_repeats):
        tik = time.time()
        fn()
        tok = time.time()
        times.append(tok - tik)
    return np.mean(times), np.var(times)


parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)
args.arch = "unet_transformer"

DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

if DEVICE == torch.device("cpu"):
    print("running on CPU")

unet_model = tasks.setup_task(args).build_model(args).to(DEVICE)
unet_encoder = unet_model.encoder

args.arch = "transformer"
args.encoder_layerdrop = False
transformer_model = tasks.setup_task(args).build_model(args).to(DEVICE)
transformer_encoder = transformer_model.encoder

BATCH_SIZE = 11
SEQ_LEN = 71

src_tokens = torch.randint(high=31, size=(BATCH_SIZE, SEQ_LEN))
src_lens = torch.tensor(BATCH_SIZE * [SEQ_LEN])
src_embs = torch.randn(size=(SEQ_LEN, BATCH_SIZE, args.encoder_embed_dim))
pad_mask = src_tokens.eq(1)


def call_model(model):
    src_tokens = torch.randint(high=31, size=(BATCH_SIZE, SEQ_LEN)).to(DEVICE)
    src_lens = torch.tensor(BATCH_SIZE * [SEQ_LEN]).to(DEVICE)
    model(src_tokens, src_lens, prev_output_tokens=src_tokens)


def call_encoder(model):
    src_tokens = torch.randint(high=31, size=(BATCH_SIZE, SEQ_LEN)).to(DEVICE)
    src_lens = torch.tensor(BATCH_SIZE * [SEQ_LEN]).to(DEVICE)
    model(src_tokens, src_lens)


def call_layer(layer):
    src_tokens = torch.randint(high=31, size=(BATCH_SIZE, SEQ_LEN)).to(DEVICE)
    src_embs = torch.randn(size=(SEQ_LEN, BATCH_SIZE, args.encoder_embed_dim)).to(DEVICE)
    pad_mask = src_tokens.eq(1)
    layer(src_embs, pad_mask)


# profiling:

_mean, _var = timeitnow(lambda: call_model(unet_model))
print(f"Unet model       : {_mean, _var}")

_mean, _var = timeitnow(lambda: call_model(transformer_model))
print(f"Transformer model: {_mean, _var}")

_mean, _var = timeitnow(lambda: call_encoder(unet_encoder))
print(f"Unet encoder       : {_mean, _var}")

_mean, _var = timeitnow(lambda: call_encoder(transformer_encoder))
print(f"Transformer encoder: {_mean, _var}")

_mean, _var = timeitnow(lambda: call_layer(unet_encoder.input_layer))
print(f"Unet input layer   : {_mean, _var}")

_mean, _var = timeitnow(lambda: call_layer(transformer_encoder.layers[0]))
print(f"Transformer layer  : {_mean, _var}")
