from generator import Generator
import argparse
import main
import torch as t

# =========================parameters about script============================
parser = argparse.ArgumentParser(description='poem para')
parser.add_argument('-word', '-w', action='store', default=None, type=int)
parser.add_argument('-v', action='store_true')
opt0 = parser.parse_args()

# =========================parameters about generator============================
VOCAB_SIZE = 5000
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 20

# =========================parameters about generator============================
generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, main.opt.cuda)
if main.opt.cuda:
    generator = generator.cuda()

# =========================generate the poem=====================================
x = t.Tensor([[5]]).long()
print(generator.sample(batch_size=20, seq_len=g_sequence_len, x=x))
