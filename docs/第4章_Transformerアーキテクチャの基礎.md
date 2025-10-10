# 第4章：Transformerアーキテクチャの基礎

この章では、現代のLLMの中核である**Transformer**を学びます。2017年に発表された"Attention is All You Need"論文で提案されたこのアーキテクチャは、自然言語処理を革命的に変えました。

---

## 4.1 アテンション機構の定式化

### 4.1.1 アテンションの直感的理解

**問題：従来のRNN/LSTMの限界**

```
長い文：「私は昨日、友達と一緒に映画館に行って、素晴らしい映画を見ました。」

RNN/LSTMの処理：
入力1 → 入力2 → ... → 入力N
  ↓      ↓            ↓
 隠れ状態が順次更新
  
問題：
- 初期の情報（「私は」）が薄れる
- 長距離依存関係の学習が困難
```

**アテンションのアイデア：**

> 「どの入力に注目すべきか」を動的に決定

**比喩：図書館での調査**

```
質問：「Transformerの発明者は？」

従来の方法：
本を順番に全部読む → 非効率

アテンション：
1. 質問に関連する本を検索（注目）
2. 重要な箇所だけ読む
3. 答えを統合
```

### 4.1.2 スケールドドット積アテンション

**基本的な設定：**

3つのベクトルを使用：
- **Query（Q）**：「何を探しているか」（質問）
- **Key（K）**：「何を持っているか」（索引）
- **Value（V）**：「実際の内容」（値）

**数式：**

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

ここで：
- $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$：Query行列
- $\mathbf{K} \in \mathbb{R}^{m \times d_k}$：Key行列
- $\mathbf{V} \in \mathbb{R}^{m \times d_v}$：Value行列
- $d_k$：Keyの次元
- $\sqrt{d_k}$：スケーリング係数

**ステップバイステップ：**

**1. 類似度の計算**

$$\mathbf{S} = \frac{\mathbf{QK}^\top}{\sqrt{d_k}} \in \mathbb{R}^{n \times m}$$

各 $S_{ij}$ は Query $i$ と Key $j$ の類似度

**2. 注目重みの計算**

$$\mathbf{A} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{n \times m}$$

各行が確率分布（合計が1）

**3. 重み付き和**

$$\text{Output} = \mathbf{AV} \in \mathbb{R}^{n \times d_v}$$

### 4.1.3 具体例（1次元）

**設定：**

文章：「猫 が 魚 を 食べる」

各単語を2次元ベクトルで表現（簡略化）：

| 単語 | ベクトル |
|------|----------|
| 猫 | $[1.0, 0.1]$ |
| が | $[0.1, 0.2]$ |
| 魚 | $[0.9, 0.2]$ |
| を | $[0.2, 0.1]$ |
| 食べる | $[0.5, 0.8]$ |

**Query：「猫」はどの単語に注目すべきか**

**ステップ1：類似度スコア**

$\mathbf{Q} = [1.0, 0.1]$（猫のベクトル）

$\mathbf{K}$：各単語のベクトル

$$S_{\text{猫}} = \frac{1}{\sqrt{2}}[1.0, 0.1] \cdot \begin{bmatrix} 1.0 \\\ 0.1 \\\ 0.9 \\\ 0.2 \\\ 0.5 \end{bmatrix}^\top \approx [0.72, 0.09, 0.65, 0.15, 0.42]$$

**ステップ2：Softmax**

$$A_{\text{猫}} = \text{softmax}([0.72, 0.09, 0.65, 0.15, 0.42])$$

$$\approx [0.35, 0.19, 0.33, 0.20, 0.27]$$

**解釈：**
- 「猫」自身に35%注目
- 「魚」に33%注目（意味的に関連）
- 「食べる」に27%注目（動作の主語）

**ステップ3：出力**

各Valueベクトルを注目重みで加重平均

$$\text{Output}_{\text{猫}} = 0.35 \times \mathbf{v}_{\text{猫}} + 0.33 \times \mathbf{v}_{\text{魚}} + \cdots$$

### 4.1.4 なぜスケーリング（ $\sqrt{d_k}$）が必要か

**問題：大きな次元での内積**

$d_k = 64$ の場合、2つのランダム単位ベクトルの内積：

$$\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{64} q_i k_i$$

期待値： $\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = 0$

分散： $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k = 64$

**結果：**

内積の値が大きく散らばる → Softmaxが飽和

**Softmax飽和の例：**

```
入力: [10, 8, 1, 0]
Softmax: [0.88, 0.12, 0.00, 0.00]  ← ほぼone-hot（勾配小）

入力: [2, 1.6, 0.2, 0]
Softmax: [0.44, 0.30, 0.15, 0.11]  ← 滑らか（勾配大）
```

**解決策： $\sqrt{d_k}$ で割る**

$$\text{Var}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

分散を1に正規化 → Softmaxの入力が適切な範囲に

### 4.1.5 Softmax関数の復習

**定義：**

$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**性質：**
1. 出力は確率分布： $\sum_i \text{softmax}(\mathbf{x})_i = 1$
2. 非負： $\text{softmax}(\mathbf{x})_i \geq 0$
3. 最大値を強調：大きな $x_i$ に高い確率

**例：**

$$\mathbf{x} = [1, 2, 3]$$

$$\text{softmax}(\mathbf{x}) = \frac{[e^1, e^2, e^3]}{e^1 + e^2 + e^3} = \frac{[2.72, 7.39, 20.09]}{30.19} \approx [0.09, 0.24, 0.67]$$

**温度パラメータ：**

$$\text{softmax}(\mathbf{x}/T)_i = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}$$

- $T > 1$：分布が平坦（多様性）
- $T < 1$：分布が鋭い（確信）
- $T \to 0$：argmaxに近づく

```
T=0.5: [0.02, 0.14, 0.84]  鋭い
T=1.0: [0.09, 0.24, 0.67]  標準
T=2.0: [0.19, 0.29, 0.52]  平坦
```

---

### 4.1.6 マルチヘッドアテンションの線形代数的解釈

**モチベーション：**

単一のアテンションでは、1種類の関係しか捉えられない

**例：**

```
文：「The cat sat on the mat」

必要な注目：
- 文法的関係（主語-動詞）
- 意味的関係（猫-マット）
- 位置関係（on）

→ 複数の「視点」が必要
```

**マルチヘッドアテンションの定義：**

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

ここで各ヘッドは：

$$\text{head}_i = \text{Attention}(\mathbf{QW}_i^Q, \mathbf{KW}_i^K, \mathbf{VW}_i^V)$$

**パラメータ：**
- $\mathbf{W}_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$：Query射影
- $\mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$：Key射影
- $\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$：Value射影
- $\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$：出力射影

通常： $d_k = d_v = d_{\text{model}}/h$

**視覚化（h=8ヘッドの場合）：**

```
入力 (d_model次元)
  ↓
分割して射影
  ├→ Head1 (d_k次元) → Attention → Output1
  ├→ Head2 (d_k次元) → Attention → Output2
  ├→ ...
  └→ Head8 (d_k次元) → Attention → Output8
       ↓
     結合 (h×d_v次元)
       ↓
    線形変換 W^O
       ↓
     出力 (d_model次元)
```

**具体的な数値例（GPT-2 Small）：**

- $d_{\text{model}} = 768$
- $h = 12$
- $d_k = d_v = 768/12 = 64$

各ヘッド：64次元で独立にアテンション計算

**なぜ効果的か：**

1. **異なる表現部分空間**

   各ヘッドが異なる特徴を学習
   
   ```
   Head1: 文法関係
   Head2: 意味関係
   Head3: 共参照
   ...
   ```

2. **計算効率**

   ```
   単一ヘッド (768次元):
   計算量: O(n² × 768²)
   
   12ヘッド (64次元×12):
   計算量: O(n² × 64² × 12) = O(n² × 768 × 64)
   
   → より効率的！
   ```

3. **並列化**

   各ヘッドは独立 → GPU並列処理可能

**実装の詳細：**

```python
# 疑似コード
def multi_head_attention(Q, K, V, num_heads):
    d_model = Q.shape[-1]
    d_k = d_model // num_heads
    
    # 射影して分割
    Q = linear(Q, W_Q).split(num_heads)  # [batch, heads, len, d_k]
    K = linear(K, W_K).split(num_heads)
    V = linear(V, W_V).split(num_heads)
    
    # 各ヘッドでアテンション
    outputs = []
    for i in range(num_heads):
        attn = attention(Q[i], K[i], V[i])
        outputs.append(attn)
    
    # 結合と射影
    concat = concatenate(outputs)  # [batch, len, d_model]
    output = linear(concat, W_O)
    
    return output
```

### 4.1.7 Self-Attention vs Cross-Attention

**Self-Attention（自己アテンション）：**

$$\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$$

同じ系列内で注目関係を計算

**用途：**
- 文章内の単語間の関係
- GPTなどのデコーダー専用モデル

**例：**
```
入力：「The cat sat on the mat」

各単語が他の単語に注目：
- "cat" → "The", "cat", "sat", ... （全単語）
- "sat" → "The", "cat", "sat", ... （全単語）
```

**Cross-Attention（交差アテンション）：**

$$\mathbf{Q} = \mathbf{X}, \quad \mathbf{K} = \mathbf{V} = \mathbf{Y}$$

異なる系列間で注目関係を計算

**用途：**
- 機械翻訳（元言語 → 目標言語）
- 画像キャプション（画像 → テキスト）

**例（翻訳）：**
```
英語（Key/Value）: "The cat sat"
日本語（Query）:   "猫が"

"猫" → "cat"に高い注目
"が" → "The"に高い注目
```

---

## 4.2 位置エンコーディングの理論

### 4.2.1 なぜ位置情報が必要か

**問題：アテンションは順序不変**

```
文1：「犬が猫を追いかけた」
文2：「猫を犬が追いかけた」
文3：「追いかけた犬が猫を」

アテンションだけでは区別できない！
（単語の集合として扱われる）
```

**必要性：**

> 単語の**位置**や**順序**を明示的に与える

### 4.2.2 絶対位置エンコーディング

#### 正弦波位置エンコーディング

**Transformerオリジナルの方法：**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

ここで：
- $pos$：位置（0, 1, 2, ...）
- $i$：次元のインデックス（0, 1, ..., $d_{\text{model}}/2-1$）

**直感的理解：**

> 各次元が異なる周波数で振動  
> → 位置を一意に識別できる「バーコード」

**視覚化（簡略版、d=4）：**

```
次元0,1: 高周波  sin(pos/10000^0)    = sin(pos)
次元2,3: 低周波  sin(pos/10000^{2/4}) = sin(pos/100)

位置  | dim0   | dim1   | dim2    | dim3
-----|--------|--------|---------|--------
0    | 0.00   | 1.00   | 0.00    | 1.00
1    | 0.84   | 0.54   | 0.01    | 1.00
2    | 0.91   |-0.42   | 0.02    | 1.00
3    | 0.14   |-0.99   | 0.03    | 1.00
```

**なぜこの形式？**

1. **有界性**： $\sin, \cos \in [-1, 1]$

2. **線形性**：

   $$PE_{pos+k} = f(PE_{pos})$$

   相対位置の計算が可能

3. **外挿性**：
   訓練時より長い系列でも機能（理論上）

**具体例（d_model=512）：**

```python
import numpy as np

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            # 偶数次元
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            # 奇数次元
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
    
    return PE

# 使用例
PE = positional_encoding(max_len=100, d_model=512)
embedding_with_pos = word_embedding + PE
```

**視覚化（ヒートマップ）：**

```
位置 →
次元
↓
   0  10  20  30  40  50
0  ▓▓░░▓▓░░▓▓░░▓▓  高周波（速く振動）
50 ▓░▓░▓░▓░▓░▓░▓░  中周波
100▓▓▓▓▓░░░░▓▓▓▓  低周波（ゆっくり振動）
```

#### 学習可能な位置埋め込み

**代替案（BERT、GPTなど）：**

位置エンコーディングを学習可能なパラメータとして扱う

$$\mathbf{PE} \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}$$

各位置に独立した埋め込みベクトル

**利点：**
- データから最適な表現を学習
- 実装が単純

**欠点：**
- 訓練時の最大長 $L_{\max}$ を超えられない
- パラメータ数増加

**GPT-2の例：**
- $L_{\max} = 1024$
- $d_{\text{model}} = 768$
- パラメータ数： $1024 \times 768 = 786,432$

### 4.2.3 相対位置エンコーディング

**問題：絶対位置の限界**

```
「猫が魚を食べる」
「犬が肉を食べる」

重要なのは「が」と「食べる」の相対位置（4単語離れている）
絶対位置（2番目、6番目）ではない
```

**相対位置エンコーディング：**

アテンションスコアに相対位置情報を追加

$$A_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j + \mathbf{q}_i \cdot \mathbf{r}_{i-j}}{\sqrt{d_k}}$$

ここで $\mathbf{r}_{i-j}$ は相対位置 $i-j$ の埋め込み

**利点：**
- 位置関係が一般化しやすい
- 長い系列への外挿が改善

### 4.2.4 回転位置エンコーディング（RoPE）

**最新の手法（GPT-Neo、LLaMAなど）：**

**アイデア：**

> ベクトルを複素平面で回転させる  
> → 位置情報を幾何学的に埋め込む

**数式（2次元の場合）：**

$$\begin{bmatrix} q_0' \\\ q_1' \end{bmatrix} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} q_0 \\\ q_1 \end{bmatrix}$$

ここで：
- $m$：位置
- $\theta$：回転角度（次元ごとに異なる）

**視覚化（2次元）：**

```
複素平面
   q₁
    ↑
    |  ●(q') 位置mでθ回転
    | /
    |/___→ q₀
   ●(q) 元のベクトル
```

**高次元への拡張：**

d次元を d/2 個の2次元部分空間に分割し、それぞれを回転

**重要な性質：**

$$\mathbf{q}_m^\top \mathbf{k}_n = \mathbf{q}_0^\top \mathbf{R}_{n-m} \mathbf{k}_0$$

内積が相対位置 $n-m$ のみに依存！

**利点：**
1. 相対位置を自然に表現
2. 長い系列への外挿が良好
3. 計算効率的

**LLaMA-2での採用：**

最新のオープンソースLLMで標準的に使用

---

## 4.3 Layer Normalizationの統計的性質

### 4.3.1 正規化の必要性

**問題：内部共変量シフト**

訓練中に各層の入力分布が変化 → 学習が不安定

**視覚化：**

```
層1の出力 → 層2の入力

エポック1: 平均=0, 分散=1
エポック2: 平均=2, 分散=4  ← 分布が変化
エポック3: 平均=-1, 分散=0.5

層2は常に変化する分布に適応する必要がある
```

### 4.3.2 Batch Normalization vs Layer Normalization

#### Batch Normalization（CNN用）

**正規化の軸：バッチ次元**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$\mu_B = \frac{1}{B}\sum_{i=1}^{B} x_i, \quad \sigma_B^2 = \frac{1}{B}\sum_{i=1}^{B}(x_i - \mu_B)^2$$

**問題点（LLM）：**

```
バッチサイズBに依存
→ 小さなバッチで性能低下
→ 系列長が異なると困難
→ 推論時にバッチ統計が必要
```

#### Layer Normalization（Transformer用）

**正規化の軸：特徴次元**

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

ここで：

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$$

- $\gamma, \beta$：学習可能なパラメータ（スケールとシフト）
- $\epsilon$：数値安定性のための小さな定数（例： $10^{-5}$）

**視覚化：**

```
入力: [batch, seq_len, d_model]

Batch Norm: バッチ全体で正規化
    ↕ (バッチ方向)

Layer Norm: 各サンプル・各位置で正規化
    ↔ (特徴方向)
```

**利点（LLM）：**
1. バッチサイズに依存しない
2. 系列長に依存しない
3. 推論が簡単

### 4.3.3 Layer Normalizationの効果

**1. 勾配の安定化**

正規化により、勾配の大きさが制御される

**2. 学習率の範囲拡大**

より大きな学習率が使用可能 → 速い収束

**3. 初期化への依存減少**

重みの初期値の影響が小さくなる

**実験的証拠：**

```
Layer Normなし:
学習率 0.001 → 収束
学習率 0.01  → 発散

Layer Normあり:
学習率 0.001 → 収束
学習率 0.01  → 収束（速い）
学習率 0.1   → 収束
```

### 4.3.4 Pre-LN vs Post-LN

**Post-LN（オリジナルTransformer）：**

```
x → Self-Attention → Add → LayerNorm
  ↘___残差接続_____↗

  → Feed-Forward → Add → LayerNorm
  ↘___残差接続___↗
```

**Pre-LN（最近のLLM）：**

```
x → LayerNorm → Self-Attention → Add
  ↘________残差接続______________↗

  → LayerNorm → Feed-Forward → Add
  ↘________残差接続____________↗
```

**Pre-LNの利点：**

1. **訓練が安定**
   - 勾配がスムーズに流れる
   - 深いモデルで特に有効

2. **Warmupが不要**
   - 学習率のウォームアップなしで訓練可能

**GPT-2以降はPre-LNが標準**

---

## 4.4 フィードフォワードネットワークの役割

### 4.4.1 FFNの構造

**定義：**

$$\text{FFN}(x) = \max(0, x\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

または GELU を使用：

$$\text{FFN}(x) = \text{GELU}(x\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

**構造：**

```
入力 (d_model)
    ↓
線形変換 W₁
    ↓
拡大 (d_ff = 4×d_model)
    ↓
活性化 (ReLU/GELU)
    ↓
線形変換 W₂
    ↓
縮小 (d_model)
    ↓
出力 (d_model)
```

**典型的なサイズ：**

- $d_{\text{model}} = 768$
- $d_{ff} = 3072 = 4 \times 768$

**パラメータ数：**

$$\text{FFN params} = d_{\text{model}} \times d_{ff} \times 2 = 768 \times 3072 \times 2 \approx 4.7M$$

（1層あたり）

### 4.4.2 FFNの役割

#### 1. 非線形変換

アテンションは線形演算（重み付き和）  
→ FFNで非線形性を導入

#### 2. 位置ごとの処理

**重要な特徴：**

> FFNは各位置に**独立に**適用される

$$\text{FFN}(\mathbf{X})_{i,:} = \text{FFN}(\mathbf{x}_i)$$

```
位置1の表現 → FFN → 変換後
位置2の表現 → FFN → 変換後
...
（同じFFN、独立に処理）
```

#### 3. 特徴変換

**解釈：**

各ニューロンが特定のパターンを検出

**例（仮想的）：**

```
ニューロン1: "過去形の動詞"を検出
ニューロン2: "複数形の名詞"を検出
ニューロン3: "否定表現"を検出
...
```

#### 4. 記憶容量

**FFNはモデルの「記憶」**

パラメータの大部分がFFNに（約2/3）

**GPT-2 Smallの例：**

- 総パラメータ：117M
- FFN：約80M（約68%）
- アテンション：約37M（約32%）

### 4.4.3 FFNの変種

#### 1. Gated FFN

**定義：**

$$\text{FFN}(x) = (\text{GELU}(x\mathbf{W}_1) \odot x\mathbf{W}_g)\mathbf{W}_2$$

ゲート機構で情報の流れを制御

#### 2. Mixture of Experts（MoE）

**アイデア：**

複数の「専門家」FFNを用意し、入力に応じて選択

$$\text{MoE}(x) = \sum_{i=1}^{n} g_i(x) \cdot \text{FFN}_i(x)$$

ここで $g_i(x)$ はゲート関数

**利点：**
- パラメータ数を増やしながら計算量を抑える
- 各専門家が特定のタスクを学習

**GPT-4で使用されていると推測される**

---

## 本章のまとめ

### 学んだこと

✅ **アテンション機構**
- Query、Key、Valueの3要素
- スケールドドット積アテンション
- Softmaxで注目重みを計算

✅ **マルチヘッドアテンション**
- 複数の表現部分空間
- 並列化による効率化
- 異なる種類の関係を捕捉

✅ **位置エンコーディング**
- 正弦波エンコーディング（周波数ベース）
- 学習可能な埋め込み
- RoPE（回転位置エンコーディング）

✅ **Layer Normalization**
- 特徴次元で正規化
- 学習を安定化
- Pre-LN vs Post-LN

✅ **フィードフォワードネットワーク**
- 位置ごとの非線形変換
- モデルの主要な記憶容量
- 4倍の拡大が標準

### 重要な公式

| 概念 | 公式 |
|------|------|
| スケールドドット積 | $\text{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^\top}{\sqrt{d_k}}\right)\mathbf{V}$ |
| マルチヘッド | $\text{MultiHead} = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\mathbf{W}^O$ |
| 正弦波位置 | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$ |
| Layer Norm | $\text{LN}(x) = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} \cdot \gamma + \beta$ |
| FFN | $\text{FFN}(x) = \text{GELU}(x\mathbf{W}_1)\mathbf{W}_2$ |

### Transformerの全体像

```
入力トークン
    ↓
トークン埋め込み + 位置エンコーディング
    ↓
┌─────────────────────┐
│  Transformer Block  │ ×N層
│  ┌──────────────┐  │
│  │ Layer Norm    │  │
│  │     ↓         │  │
│  │ Multi-Head    │  │
│  │ Attention     │  │
│  │     ↓         │  │
│  │ 残差接続      │  │
│  └──────────────┘  │
│  ┌──────────────┐  │
│  │ Layer Norm    │  │
│  │     ↓         │  │
│  │ Feed-Forward  │  │
│  │     ↓         │  │
│  │ 残差接続      │  │
│  └──────────────┘  │
└─────────────────────┘
    ↓
出力
```

### 次章の予告

第5章では、Transformerを使った**自己回帰言語モデル**を学びます：
- 自己回帰モデリングの確率論的基礎
- 因果的マスキング
- 次トークン予測
- 文脈長と記憶容量

GPTのような生成モデルがどのように動作するかを理解していきます。

---

## 練習問題

### 問題1：アテンションスコアの計算

$$\mathbf{Q} = [1, 0], \mathbf{K} = \begin{bmatrix} 1 & 0 \\\ 0 & 1 \\\ 1 & 1 \end{bmatrix}, d_k=2$$

のとき、スケール後のスコアを計算せよ。

### 問題2：Softmax
$\mathbf{x} = [1, 2, 3, 4]$ のSoftmaxを計算せよ。

### 問題3：位置エンコーディング
$pos=0, i=0, d_{\text{model}}=512$ のとき、 $PE_{(0,0)}$ と $PE_{(0,1)}$ を計算せよ。

### 問題4：Layer Normalization
$\mathbf{x} = [1, 2, 3, 4]$, $\gamma=1$, $\beta=0$, $\epsilon=0$ として Layer Norm の出力を計算せよ。

### 解答

**問題1:**

$$\mathbf{S} = \frac{1}{\sqrt{2}}\mathbf{QK}^\top = \frac{1}{\sqrt{2}}[1, 0]\begin{bmatrix} 1 & 0 & 1 \\\ 0 & 1 & 1 \end{bmatrix} = \frac{1}{\sqrt{2}}[1, 0, 1] \approx [0.707, 0, 0.707]$$

**問題2:**

$$e^{\mathbf{x}} = [e^1, e^2, e^3, e^4] = [2.72, 7.39, 20.09, 54.60]$$

$$\sum = 84.79$$

$$\text{softmax}(\mathbf{x}) = [0.032, 0.087, 0.237, 0.644]$$

**問題3:**

$$PE_{(0,0)} = \sin(0/10000^0) = \sin(0) = 0$$

$$PE_{(0,1)} = \cos(0/10000^0) = \cos(0) = 1$$

**問題4:**

$$\mu = (1+2+3+4)/4 = 2.5$$

$$\sigma^2 = ((1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2)/4 = 1.25$$

$$\text{LN}(\mathbf{x}) = \frac{[1,2,3,4]-2.5}{\sqrt{1.25}} = \frac{[-1.5,-0.5,0.5,1.5]}{1.118} \approx [-1.34, -0.45, 0.45, 1.34]$$

---

**📖 前章：[第3章 ニューラルネットワークの基礎](./第3章_ニューラルネットワークの基礎.md)**  
**📖 次章：[第5章 自己回帰言語モデルの理論](./第5章_自己回帰言語モデルの理論.md)**
