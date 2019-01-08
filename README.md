# General-Transformer-Pytorch
Implementation of Transformer for General Usage, like encoding context from sequence input.

Most of the codes are from Ref.2.

**TEST ON: Python 3.6, Pytorch 0.4.1**

# Mask
Looks like:
```
[
  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
  ...
]
```

# Demo
```
from Transformer import TransformerBlock
import numpy as np

sample_size = 100
max_len = 10
hidden = 256
attn_heads = 4
dropout = 0.1
transformer = TransformerBlock(hidden, attn_heads, hidden * 4, dropout, is_cuda=False)

x = np.random.rand(sample_size, max_len, hidden)
x = torch.Tensor(x)
output = transformer.forward(x, mask=None)
print(output.size()
```

# Reference
1. Transformer: https://arxiv.org/abs/1706.03762
2. Bert-Pytorch: https://github.com/codertimo/BERT-pytorch
