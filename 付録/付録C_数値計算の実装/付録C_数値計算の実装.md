# 付録C：数値計算の実装

理論を実践に移すためのPython実装例を提供します。

---

## C.1 基本的な数学関数

### C.1.1 活性化関数

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """シグモイド関数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """シグモイドの導関数"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU関数"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLUの導関数"""
    return (x > 0).astype(float)

def gelu(x):
    """GELU関数（近似版）"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_exact(x):
    """GELU関数（正確版）"""
    from scipy.stats import norm
    return x * norm.cdf(x)

def softmax(x, axis=-1):
    """Softmax関数（数値安定版）"""
    # オーバーフロー防止
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 可視化
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, np.tanh(x), label='Tanh')
plt.legend()
plt.grid(True)
plt.title('Sigmoid & Tanh')

plt.subplot(1, 3, 2)
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, np.maximum(0, x) * np.minimum(1, x/6 + 0.5), label='HardSwish')
plt.legend()
plt.grid(True)
plt.title('ReLU & Variants')

plt.subplot(1, 3, 3)
plt.plot(x, gelu(x), label='GELU')
plt.plot(x, x * sigmoid(x), label='Swish/SiLU')
plt.legend()
plt.grid(True)
plt.title('Modern Activations')

plt.tight_layout()
# plt.savefig('activations.png')
plt.show()
```

### C.1.2 損失関数

```python
def cross_entropy(y_true, y_pred, epsilon=1e-12):
    """
    クロスエントロピー損失
    
    Args:
        y_true: 真のラベル (one-hot or labels)
        y_pred: 予測確率
        epsilon: 数値安定化
    """
    # クリッピングで log(0) を防止
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    if len(y_true.shape) == 1:  # ラベル形式
        n_samples = y_true.shape[0]
        return -np.sum(np.log(y_pred[np.arange(n_samples), y_true])) / n_samples
    else:  # one-hot形式
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def mse_loss(y_true, y_pred):
    """平均二乗誤差"""
    return np.mean((y_true - y_pred)**2)

def perplexity(cross_entropy_loss):
    """パープレキシティ"""
    return np.exp(cross_entropy_loss)

# 例
y_true = np.array([1, 0, 2])  # クラスラベル
y_pred = np.array([
    [0.1, 0.7, 0.2],  # サンプル1の予測
    [0.8, 0.1, 0.1],  # サンプル2の予測
    [0.1, 0.2, 0.7]   # サンプル3の予測
])

loss = cross_entropy(y_true, y_pred)
ppl = perplexity(loss)
print(f"Cross Entropy: {loss:.4f}")
print(f"Perplexity: {ppl:.4f}")
```

---

## C.2 線形代数の実装

### C.2.1 行列演算

```python
def matrix_multiply(A, B):
    """行列積（教育用・低速）"""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "次元が合いません"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

def batch_matrix_multiply(A, B):
    """バッチ行列積"""
    # A: (batch, m, n)
    # B: (batch, n, p)
    # 出力: (batch, m, p)
    return np.einsum('bmn,bnp->bmp', A, B)

# NumPy の高速実装を使う方が実用的
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = A @ B  # Python 3.5+
# または
C = np.dot(A, B)
```

### C.2.2 SVD と低ランク近似

```python
def low_rank_approximation(A, k):
    """
    行列の低ランク近似（LoRAの基礎）
    
    Args:
        A: 入力行列 (m, n)
        k: ランク
    
    Returns:
        A_k: ランクk近似
        U_k, S_k, Vt_k: SVD成分
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # 上位k個の特異値のみ使用
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # 再構成
    A_k = U_k @ np.diag(S_k) @ Vt_k
    
    return A_k, U_k, S_k, Vt_k

# 例：画像の圧縮
from PIL import Image

# ダミー画像（実際にはImage.open()で読み込み）
img_array = np.random.rand(100, 100)

# 様々なランクで近似
ranks = [5, 10, 20, 50]
fig, axes = plt.subplots(1, len(ranks)+1, figsize=(15, 3))

axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for idx, k in enumerate(ranks):
    A_k, _, _, _ = low_rank_approximation(img_array, k)
    axes[idx+1].imshow(A_k, cmap='gray')
    axes[idx+1].set_title(f'Rank {k}')
    axes[idx+1].axis('off')

plt.tight_layout()
# plt.savefig('svd_compression.png')
plt.show()
```

### C.2.3 ノルムと距離

```python
def compute_norms(v):
    """各種ノルムの計算"""
    l1 = np.linalg.norm(v, ord=1)  # L1ノルム
    l2 = np.linalg.norm(v, ord=2)  # L2ノルム
    linf = np.linalg.norm(v, ord=np.inf)  # L∞ノルム
    
    return {
        'L1': l1,
        'L2': l2,
        'L_inf': linf
    }

def cosine_similarity(a, b):
    """コサイン類似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """ユークリッド距離"""
    return np.linalg.norm(a - b)

# 例
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print("Norms of v1:", compute_norms(v1))
print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")
print(f"Euclidean distance: {euclidean_distance(v1, v2):.4f}")
```

---

## C.3 Attention機構の実装

### C.3.1 Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention
    
    Args:
        Q: Query (batch, seq_len, d_k)
        K: Key (batch, seq_len, d_k)
        V: Value (batch, seq_len, d_v)
        mask: マスク (batch, seq_len, seq_len) または None
    
    Returns:
        output: (batch, seq_len, d_v)
        attention_weights: (batch, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # スコア計算: (batch, seq_len, seq_len)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # マスク適用
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Softmax
    attention_weights = softmax(scores, axis=-1)
    
    # 重み付き和
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

# 例
batch_size = 2
seq_len = 4
d_k = 8
d_v = 8

Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_v)

# Causal mask（自己回帰用）
causal_mask = np.tril(np.ones((seq_len, seq_len)))
causal_mask = causal_mask[np.newaxis, :, :]  # (1, seq_len, seq_len)

output, attn_weights = scaled_dot_product_attention(Q, K, V, causal_mask)

print("Output shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)
print("\nAttention pattern (sample 0):")
print(attn_weights[0])
```

### C.3.2 Multi-Head Attention

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        Multi-Head Attention
        
        Args:
            d_model: モデル次元
            num_heads: ヘッド数
        """
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 重み行列（簡略化のためランダム初期化）
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x):
        """(batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)"""
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        """(batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)"""
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) または None
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = x @ self.W_q  # (batch, seq_len, d_model)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention for each head
        d_k = self.d_k
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        if mask is not None:
            # Broadcast mask for all heads
            mask = mask[:, np.newaxis, :, :]
            scores = np.where(mask == 0, -1e9, scores)
        
        attn_weights = softmax(scores, axis=-1)
        attn_output = np.matmul(attn_weights, V)
        
        # Combine heads
        attn_output = self.combine_heads(attn_output)
        
        # Final linear projection
        output = attn_output @ self.W_o
        
        return output

# 例
d_model = 512
num_heads = 8
batch_size = 2
seq_len = 10

mha = MultiHeadAttention(d_model, num_heads)
x = np.random.randn(batch_size, seq_len, d_model)

output = mha.forward(x)
print("Multi-Head Attention output shape:", output.shape)
```

---

## C.4 位置エンコーディング

### C.4.1 正弦波位置エンコーディング

```python
def positional_encoding(max_len, d_model):
    """
    正弦波位置エンコーディング
    
    Args:
        max_len: 最大系列長
        d_model: モデル次元
    
    Returns:
        pe: (max_len, d_model)
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# 可視化
max_len = 100
d_model = 128

pe = positional_encoding(max_len, d_model)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(pe, cmap='RdBu', aspect='auto')
plt.colorbar()
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding')

plt.subplot(1, 2, 2)
# いくつかの次元をプロット
for i in [0, 1, 10, 50]:
    plt.plot(pe[:, i], label=f'dim {i}')
plt.xlabel('Position')
plt.ylabel('Value')
plt.legend()
plt.title('PE by Dimension')
plt.grid(True)

plt.tight_layout()
# plt.savefig('positional_encoding.png')
plt.show()
```

### C.4.2 回転位置エンコーディング（RoPE）

```python
def rope(q, k, positions, d_model):
    """
    Rotary Position Embedding (RoPE)
    
    Args:
        q, k: Query, Key vectors (batch, seq_len, d_model)
        positions: 位置インデックス (seq_len,)
        d_model: モデル次元
    
    Returns:
        q_rope, k_rope: 回転適用後
    """
    def rotate_half(x):
        # (batch, seq_len, d_model) → 前半と後半を入れ替え
        x1 = x[..., :d_model//2]
        x2 = x[..., d_model//2:]
        return np.concatenate([-x2, x1], axis=-1)
    
    # 周波数計算
    inv_freq = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))
    
    # 位置ごとの角度
    angles = np.outer(positions, inv_freq)  # (seq_len, d_model/2)
    
    # cos, sin
    cos = np.cos(angles)  # (seq_len, d_model/2)
    sin = np.sin(angles)
    
    # ブロードキャスト用に拡張
    cos = np.repeat(cos, 2, axis=-1)[np.newaxis, :, :]  # (1, seq_len, d_model)
    sin = np.repeat(sin, 2, axis=-1)[np.newaxis, :, :]
    
    # 回転適用
    q_rope = q * cos + rotate_half(q) * sin
    k_rope = k * cos + rotate_half(k) * sin
    
    return q_rope, k_rope

# 例
batch = 1
seq_len = 10
d_model = 64

q = np.random.randn(batch, seq_len, d_model)
k = np.random.randn(batch, seq_len, d_model)
positions = np.arange(seq_len)

q_rope, k_rope = rope(q, k, positions, d_model)
print("RoPE applied Q shape:", q_rope.shape)
```

---

## C.5 最適化アルゴリズム

### C.5.1 勾配降下法の変種

```python
class Optimizer:
    """基底クラス"""
    def __init__(self, learning_rate):
        self.lr = learning_rate
    
    def update(self, params, grads):
        raise NotImplementedError

class SGD(Optimizer):
    """確率的勾配降下法"""
    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad

class Momentum(Optimizer):
    """Momentum SGD"""
    def __init__(self, learning_rate, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grad
            param += self.v[i]

class Adam(Optimizer):
    """Adam optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)
            
            param -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)

# テスト: 簡単な最適化問題
def rosenbrock(x):
    """Rosenbrock関数（最小値は (1,1) で 0）"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    """Rosenbrock関数の勾配"""
    grad = np.zeros_like(x)
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

# 各最適化手法で比較
x_init = np.array([-1.0, -1.0])
optimizers = {
    'SGD': SGD(learning_rate=0.001),
    'Momentum': Momentum(learning_rate=0.001, momentum=0.9),
    'Adam': Adam(learning_rate=0.01)
}

histories = {}
for name, opt in optimizers.items():
    x = x_init.copy()
    history = [x.copy()]
    
    for _ in range(1000):
        grad = rosenbrock_grad(x)
        opt.update([x], [grad])
        history.append(x.copy())
    
    histories[name] = np.array(history)

# 可視化
plt.figure(figsize=(12, 4))
for name, history in histories.items():
    plt.plot(history[:, 0], history[:, 1], label=name, alpha=0.7)

plt.plot(1, 1, 'r*', markersize=15, label='Optimum')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.legend()
plt.title('Optimization Paths on Rosenbrock Function')
plt.grid(True)
plt.tight_layout()
# plt.savefig('optimizer_comparison.png')
plt.show()
```

### C.5.2 学習率スケジューリング

```python
def linear_warmup_cosine_decay(step, warmup_steps, total_steps, lr_max, lr_min=0):
    """
    Warmup + Cosine Decay
    
    Args:
        step: 現在のステップ
        warmup_steps: ウォームアップステップ数
        total_steps: 総ステップ数
        lr_max: 最大学習率
        lr_min: 最小学習率
    
    Returns:
        lr: 学習率
    """
    if step < warmup_steps:
        # Linear warmup
        return lr_max * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

# 可視化
total_steps = 10000
warmup_steps = 1000
lr_max = 0.001
lr_min = 0.00001

steps = np.arange(total_steps)
lrs = [linear_warmup_cosine_decay(s, warmup_steps, total_steps, lr_max, lr_min) 
       for s in steps]

plt.figure(figsize=(10, 4))
plt.plot(steps, lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Warmup + Cosine Decay Schedule')
plt.grid(True)
plt.tight_layout()
# plt.savefig('lr_schedule.png')
plt.show()
```

---

## C.6 トークン化

### C.6.1 簡単なBPE実装

```python
from collections import Counter, defaultdict
import re

class SimpleBPE:
    """Byte Pair Encoding（簡略版）"""
    
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.merges = {}
        self.vocab = set()
    
    def get_stats(self, word_freqs):
        """バイグラムの頻度を計算"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_pair(self, pair, word_freqs):
        """ペアをマージ"""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        
        return new_word_freqs
    
    def train(self, texts):
        """BPE学習"""
        # 文字レベルに分割
        word_freqs = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                word = ' '.join(list(word)) + ' </w>'
                word_freqs[word] += 1
        
        # 初期語彙
        self.vocab = set()
        for word in word_freqs.keys():
            self.vocab.update(word.split())
        
        # BPEマージ
        for i in range(self.num_merges):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self.merge_pair(best_pair, word_freqs)
            self.merges[best_pair] = i
            self.vocab.add(''.join(best_pair))
            
            if i % 20 == 0:
                print(f"Merge {i}: {best_pair} (freq: {pairs[best_pair]})")
        
        print(f"\nVocabulary size: {len(self.vocab)}")
    
    def tokenize(self, text):
        """テキストをトークン化（簡略版）"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            
            # マージ適用
            for pair in sorted(self.merges.keys(), key=lambda x: self.merges[x]):
                bigram = ' '.join(pair)
                if bigram in word:
                    word = word.replace(bigram, ''.join(pair))
            
            tokens.extend(word.split())
        
        return tokens

# 例
texts = [
    "the quick brown fox jumps over the lazy dog",
    "the dog was lazy",
    "the fox was quick",
] * 10  # 繰り返して頻度を上げる

bpe = SimpleBPE(num_merges=50)
bpe.train(texts)

test_text = "the quickest fox"
tokens = bpe.tokenize(test_text)
print(f"\nTokenization of '{test_text}':")
print(tokens)
```

---

## C.7 評価指標

### C.7.1 BLEU スコア

```python
from collections import Counter
import numpy as np

def ngrams(tokens, n):
    """n-gramを生成"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def bleu_score(reference, candidate, max_n=4):
    """
    BLEU スコア計算
    
    Args:
        reference: 参照文（トークンリスト）
        candidate: 候補文（トークンリスト）
        max_n: 最大n-gram
    
    Returns:
        BLEU score
    """
    # Brevity penalty
    ref_len = len(reference)
    cand_len = len(candidate)
    
    if cand_len > ref_len:
        bp = 1
    else:
        bp = np.exp(1 - ref_len / cand_len)
    
    # n-gram precision
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(reference, n))
        cand_ngrams = Counter(ngrams(candidate, n))
        
        # クリッピングされた一致数
        clipped_count = sum(min(cand_ngrams[ng], ref_ngrams[ng]) 
                           for ng in cand_ngrams)
        total_count = sum(cand_ngrams.values())
        
        if total_count == 0:
            precisions.append(0)
        else:
            precisions.append(clipped_count / total_count)
    
    # 幾何平均
    if min(precisions) > 0:
        geo_mean = np.exp(np.mean(np.log(precisions)))
    else:
        geo_mean = 0
    
    return bp * geo_mean

# 例
reference = "the cat is on the mat".split()
candidate1 = "the cat is on the mat".split()  # 完全一致
candidate2 = "the cat on the mat".split()      # 1語欠落
candidate3 = "there is a cat on the mat".split()  # 異なる

print(f"BLEU (perfect): {bleu_score(reference, candidate1):.4f}")
print(f"BLEU (missing word): {bleu_score(reference, candidate2):.4f}")
print(f"BLEU (different): {bleu_score(reference, candidate3):.4f}")
```

### C.7.2 パープレキシティ

```python
def compute_perplexity(model_probs, true_labels):
    """
    パープレキシティ計算
    
    Args:
        model_probs: モデルの予測確率 (n_samples, vocab_size)
        true_labels: 正解ラベル (n_samples,)
    
    Returns:
        perplexity
    """
    n_samples = len(true_labels)
    log_probs = np.log(model_probs[np.arange(n_samples), true_labels] + 1e-10)
    avg_log_prob = np.mean(log_probs)
    perplexity = np.exp(-avg_log_prob)
    return perplexity

# 例
vocab_size = 10000
n_samples = 100

# ダミーデータ
model_probs = softmax(np.random.randn(n_samples, vocab_size), axis=1)
true_labels = np.random.randint(0, vocab_size, n_samples)

ppl = compute_perplexity(model_probs, true_labels)
print(f"Perplexity: {ppl:.2f}")
```

---

## まとめ

この付録では、LLMに関連する数値計算の実装例を提供しました。

**実装した内容：**

✅ **基本関数**: 活性化関数、損失関数  
✅ **線形代数**: 行列演算、SVD、ノルム  
✅ **Attention**: Scaled Dot-Product、Multi-Head  
✅ **位置エンコーディング**: 正弦波、RoPE  
✅ **最適化**: SGD、Momentum、Adam、学習率スケジューリング  
✅ **トークン化**: BPE  
✅ **評価指標**: BLEU、パープレキシティ

**実用的な実装には：**

本付録のコードは教育目的です。実用にはPyTorch/TensorFlow/JAXなどのフレームワークを使用してください。

**さらに学ぶには：**

- HuggingFace Transformers ライブラリ
- PyTorch公式チュートリアル
- 各モデルの公式実装

---

**📖 前の付録：[付録B 重要定理と証明](../付録B_重要定理と証明/付録B_重要定理と証明.md)**  
**📖 次の付録：[付録D ベンチマークデータセット](../付録D_ベンチマークデータセット/付録D_ベンチマークデータセット.md)**
