# 付録B：重要定理と証明

本ガイドブックで登場した重要な定理とその証明をまとめます。

---

## B.1 基礎数学

### B.1.1 Jensen の不等式

**定理：**

$f$ を凸関数とする。このとき、

$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

（ $f$ が凹関数なら不等号が逆）

**証明（離散版）：**

凸関数の定義より、任意の $\lambda \in [0, 1]$ に対して

$$f(\lambda x_1 + (1-\lambda)x_2) \leq \lambda f(x_1) + (1-\lambda)f(x_2)$$

数学的帰納法で $n$ 個の点に拡張：

$$f\left(\sum_{i=1}^n p_i x_i\right) \leq \sum_{i=1}^n p_i f(x_i)$$

ここで $\sum p_i = 1$、$p_i \geq 0$

確率分布 $P(X=x_i) = p_i$ とすれば

$$f(\mathbb{E}[X]) = f\left(\sum_i p_i x_i\right) \leq \sum_i p_i f(x_i) = \mathbb{E}[f(X)]$$

**応用例：**

1. **KL ダイバージェンスの非負性**
   
   $$D_{\text{KL}}(P \| Q) = \mathbb{E}_P\left[\log \frac{P(x)}{Q(x)}\right]$$
   
   $-\log$ は凸関数なので、Jensen の不等式より
   
   $$-\log \mathbb{E}_P\left[\frac{Q(x)}{P(x)}\right] \leq \mathbb{E}_P\left[-\log \frac{Q(x)}{P(x)}\right] = D_{\text{KL}}(P \| Q)$$
   
   左辺：
   
   $$-\log \sum_x P(x) \frac{Q(x)}{P(x)} = -\log \sum_x Q(x) = -\log 1 = 0$$
   
   ∴ $D_{\text{KL}}(P \| Q) \geq 0$

2. **対数和不等式**
   
   $$\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$$

### B.1.2 Cauchy-Schwarz の不等式

**定理：**

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \|\mathbf{v}\|$$

等号成立条件： $\mathbf{u} = c\mathbf{v}$（平行）

**証明：**

任意の $t \in \mathbb{R}$ に対して

$$0 \leq \|\mathbf{u} - t\mathbf{v}\|^2 = \langle \mathbf{u} - t\mathbf{v}, \mathbf{u} - t\mathbf{v} \rangle$$

展開すると

$$= \|\mathbf{u}\|^2 - 2t\langle \mathbf{u}, \mathbf{v} \rangle + t^2 \|\mathbf{v}\|^2$$

これは $t$ の2次関数で常に非負。判別式 $D \leq 0$：

$$D = 4\langle \mathbf{u}, \mathbf{v} \rangle^2 - 4\|\mathbf{u}\|^2 \|\mathbf{v}\|^2 \leq 0$$

$$\Rightarrow \langle \mathbf{u}, \mathbf{v} \rangle^2 \leq \|\mathbf{u}\|^2 \|\mathbf{v}\|^2$$

両辺の平方根を取れば結果を得る。

---

## B.2 確率論

### B.2.1 大数の法則

**定理（弱法則）：**

$X_1, X_2, ...$ を独立同分布（i.i.d.）な確率変数、$\mathbb{E}[X_i] = \mu$ とする。このとき、任意の $\epsilon > 0$ に対して

$$\lim_{n \to \infty} P\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mu\right| > \epsilon\right) = 0$$

**証明の概略（Chebyshevの不等式を用いて）：**

標本平均 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$

$$\mathbb{E}[\bar{X}_n] = \mu$$

$$\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}$$

Chebyshevの不等式：

$$P(|\bar{X}_n - \mu| > \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2}$$

$n \to \infty$ で右辺 $\to 0$

**LLMへの応用：**

訓練データが十分大きければ、経験分布が真の分布に収束

### B.2.2 中心極限定理

**定理：**

$X_1, X_2, ...$ を i.i.d.、$\mathbb{E}[X_i] = \mu$、$\text{Var}(X_i) = \sigma^2$ とする。このとき

$$\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1)$$

（ $\xrightarrow{d}$ は分布収束）

**意味：**

標本平均の分布は、 $n$ が大きいとき正規分布に近づく

**LLMへの応用：**

- 勾配推定（ミニバッチ）
- モンテカルロ推定

---

## B.3 最適化理論

### B.3.1 凸関数の1階条件

**定理：**

$f$ が凸関数で微分可能なとき、

$$x^* = \arg\min_x f(x) \Leftrightarrow \nabla f(x^*) = 0$$

**証明：**

( $\Rightarrow$) $x^*$ が最小点なら、任意の方向 $d$ に対して

$$f(x^* + \epsilon d) \geq f(x^*)$$

1次のTaylor展開：

$$f(x^* + \epsilon d) \approx f(x^*) + \epsilon \langle \nabla f(x^*), d \rangle$$

$\epsilon > 0$ で上式が成立するには $\langle \nabla f(x^*), d \rangle \geq 0$

$\epsilon < 0$ でも成立するには $\langle \nabla f(x^*), d \rangle \leq 0$

両方満たすには $\langle \nabla f(x^*), d \rangle = 0$ for all $d$

∴ $\nabla f(x^*) = 0$

( $\Leftarrow$) 凸関数の性質より、任意の $x$ に対して

$$f(x) \geq f(x^*) + \langle \nabla f(x^*), x - x^* \rangle = f(x^*)$$

（ $\nabla f(x^*) = 0$ を使用）

∴ $x^*$ は最小点

**LLMへの応用：**

訓練損失（非凸だが）の臨界点を探す

### B.3.2 勾配降下法の収束（凸関数）

**定理：**

$f$ を $L$-Lipschitz連続な勾配を持つ凸関数とする。学習率 $\eta \leq \frac{1}{L}$ の勾配降下法

$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

は、 $T$ ステップ後に

$$f(\bar{x}_T) - f(x^*) \leq \frac{\|x_0 - x^*\|^2}{2\eta T}$$

を満たす。ここで $\bar{x}_T = \frac{1}{T}\sum_{t=1}^T x_t$、$x^*$ は最適解。

**証明の概略：**

$L$-smooth性より

$$f(x_{t+1}) \leq f(x_t) + \langle \nabla f(x_t), x_{t+1} - x_t \rangle + \frac{L}{2}\|x_{t+1} - x_t\|^2$$

$x_{t+1} - x_t = -\eta \nabla f(x_t)$ を代入：

$$f(x_{t+1}) \leq f(x_t) - \eta \|\nabla f(x_t)\|^2 + \frac{L\eta^2}{2}\|\nabla f(x_t)\|^2$$

$\eta \leq \frac{1}{L}$ なら

$$f(x_{t+1}) \leq f(x_t) - \frac{\eta}{2}\|\nabla f(x_t)\|^2$$

凸性と代数的操作から最終結果を導出。

**収束率：**

$$O(1/T)$$

---

## B.4 情報理論

### B.4.1 Gibbs の不等式

**定理：**

任意の2つの確率分布 $P, Q$ に対して

$$H(P) \leq H(P, Q)$$

ここで $H(P) = -\sum_x P(x) \log P(x)$（エントロピー）、$H(P, Q) = -\sum_x P(x) \log Q(x)$（クロスエントロピー）

等号成立条件： $P = Q$

**証明：**

$$H(P, Q) - H(P) = -\sum_x P(x) \log Q(x) + \sum_x P(x) \log P(x)$$

$$= \sum_x P(x) \log \frac{P(x)}{Q(x)} = D_{\text{KL}}(P \| Q) \geq 0$$

（KL ダイバージェンスの非負性より）

**LLMへの応用：**

訓練時の目的関数（クロスエントロピー最小化）は、真の分布とモデル分布を近づける

### B.4.2 データ処理不等式

**定理：**

マルコフ連鎖 $X \to Y \to Z$ に対して

$$I(X; Z) \leq I(X; Y)$$

**意味：**

情報は処理によって増加しない

**LLMへの応用：**

深い層での情報のボトルネック

---

## B.5 線形代数

### B.5.1 特異値分解（SVD）の存在

**定理：**

任意の行列 $A \in \mathbb{R}^{m \times n}$ に対して、

$$A = U\Sigma V^T$$

と分解できる。ここで
- $U \in \mathbb{R}^{m \times m}$：直交行列
- $\Sigma \in \mathbb{R}^{m \times n}$：対角行列（特異値）
- $V \in \mathbb{R}^{n \times n}$：直交行列

**証明の概略：**

1. $A^T A$ は対称行列なので、直交対角化可能：
   $$A^T A = V \Lambda V^T$$

2. $\Lambda$ の対角成分 $\lambda_i \geq 0$（半正定値性）

3. $\sigma_i = \sqrt{\lambda_i}$ とし、$u_i = \frac{1}{\sigma_i} Av_i$

4. これらが直交系をなすことを示す

**LLMへの応用：**

- 行列の低ランク近似（LoRA）
- 主成分分析（PCA）

### B.5.2 Eckart-Young定理（低ランク近似）

**定理：**

行列 $A$ のランク $k$ 近似として、フロベニウスノルムで最良のものは、SVDの上位 $k$ 特異値・特異ベクトルで構成される：

$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$$

このとき

$$\min_{\text{rank}(B) \leq k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

**LLMへの応用：**

LoRA（Low-Rank Adaptation）の理論的基盤

---

## B.6 ニューラルネットワーク理論

### B.6.1 Universal Approximation Theorem

**定理（簡略版）：**

1層の隠れ層を持つニューラルネットワークは、コンパクト集合上の連続関数を任意の精度で近似できる。

具体的には、活性化関数 $\sigma$ が非多項式なら、任意の連続関数 $f: [0,1]^n \to \mathbb{R}$ と $\epsilon > 0$ に対して、十分大きい $m$ で

$$\left\|f(x) - \sum_{i=1}^m w_i \sigma(v_i^T x + b_i)\right\| < \epsilon$$

を満たす重み $w_i, v_i, b_i$ が存在する。

**証明の概略（省略）：**

Stone-Weierstrassの定理を応用

**含意：**

- 幅が十分あれば1層で十分（理論上）
- 実際は深いネットワークの方が効率的

### B.6.2 Backpropagation の導出

**設定：**

損失 $\mathcal{L}$、層 $l$ の出力 $h^{(l)}$

$$h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$$

**目標：**

$\frac{\partial \mathcal{L}}{\partial W^{(l)}}$ を計算

**連鎖律：**

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial h^{(l)}} \frac{\partial h^{(l)}}{\partial W^{(l)}}$$

**再帰的定義：**

$$\delta^{(l)} := \frac{\partial \mathcal{L}}{\partial h^{(l)}}$$

$$\delta^{(l-1)} = \delta^{(l)} \frac{\partial h^{(l)}}{\partial h^{(l-1)}} = \delta^{(l)} \odot \sigma'(z^{(l)}) \cdot W^{(l)}$$

（ $\odot$ は要素ごとの積）

**勾配：**

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T$$

$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

**アルゴリズム：**

1. **Forward Pass**: $h^{(l)}$ を順に計算
2. **Backward Pass**: $\delta^{(L)}, \delta^{(L-1)}, ...$ を逆順に計算
3. **勾配計算**: 各層の勾配を求める

---

## B.7 Transformer理論

### B.7.1 Attention is All You Need の主要結果

**Self-Attentionの計算複雑性：**

**命題：**

系列長 $n$、次元 $d$ のSelf-Attentionの計算量は

$$O(n^2 d)$$

**証明：**

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

1. $QK^T$: $(n \times d) \times (d \times n) = O(n^2 d)$
2. softmax: $O(n^2)$（各行 $O(n)$、$n$ 行）
3. $AV$: $(n \times n) \times (n \times d) = O(n^2 d)$

総計： $O(n^2 d)$

### B.7.2 Positional Encodingの性質

**定理：**

正弦波位置エンコーディング

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

は、相対位置 $k = pos' - pos$ の関数として線形変換で表現できる。

**証明：**

三角関数の加法定理より

$$\sin(pos' / \omega) = \sin(pos/\omega) \cos(k/\omega) + \cos(pos/\omega) \sin(k/\omega)$$

$$\cos(pos' / \omega) = \cos(pos/\omega) \cos(k/\omega) - \sin(pos/\omega) \sin(k/\omega)$$

行列形式：

$$\begin{pmatrix} \sin(pos'/\omega) \\ \cos(pos'/\omega) \end{pmatrix} = \begin{pmatrix} \cos(k/\omega) & \sin(k/\omega) \\ -\sin(k/\omega) & \cos(k/\omega) \end{pmatrix} \begin{pmatrix} \sin(pos/\omega) \\ \cos(pos/\omega) \end{pmatrix}$$

回転行列による線形変換！

**含意：**

相対位置情報を保持できる

---

## B.8 スケーリング則

### B.8.1 べき乗則の導出（簡略版）

**観察：**

損失 $L$ とパラメータ数 $N$ の関係

$$L(N) \approx \left(\frac{N_c}{N}\right)^\alpha$$

**対数を取ると：**

$$\log L(N) \approx \alpha \log N_c - \alpha \log N$$

線形関係！

**実証的フィッティング：**

データに対して最小二乗法で $\alpha, N_c$ を推定

**理論的説明（仮説）：**

- 臨界現象（物理学）との類似
- 高次元の幾何学的性質
- 情報圧縮の効率

---

## B.9 強化学習理論

### B.9.1 ベルマン方程式

**定理：**

最適価値関数 $V^*(s)$ は以下を満たす：

$$V^*(s) = \max_a \left[R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]$$

**証明：**

最適方策 $\pi^*$ の下で

$$V^*(s) = \mathbb{E}_{\pi^*}\left[\sum_{t=0}^\infty \gamma^t R_t \mid S_0 = s\right]$$

$$= \mathbb{E}_{\pi^*}\left[R_0 + \gamma \sum_{t=1}^\infty \gamma^{t-1} R_t \mid S_0 = s\right]$$

$$= \max_a \left[R(s, a) + \gamma \mathbb{E}[V^*(S_1) | S_0 = s, A_0 = a]\right]$$

$$= \max_a \left[R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]$$

**LLMへの応用：**

RLHFでの価値関数学習

### B.9.2 Policy Gradient定理

**定理：**

方策 $\pi_\theta$ のパラメータ $\theta$ に関する期待リターンの勾配は

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a)\right]$$

**証明の概略：**

期待リターン：

$$J(\theta) = \mathbb{E}_{s \sim d_\pi, a \sim \pi_\theta}[R(s, a)]$$

勾配：

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_s d_\pi(s) \sum_a \pi_\theta(a|s) Q^\pi(s, a)$$

$d_\pi$ の $\theta$ 依存性は複雑だが、Policy Gradient定理により簡略化できる。

**LLMへの応用：**

PPOの理論的基盤

---

## B.10 汎化理論

### B.10.1 VC次元による汎化誤差の上界

**定理（Vapnik-Chervonenkis）：**

VC次元が $d$ の仮説クラス $\mathcal{H}$ に対して、$n$ 個のサンプルで学習した仮説 $h$ の汎化誤差は、高確率で

$$\text{Error}_{\text{test}}(h) \leq \text{Error}_{\text{train}}(h) + O\left(\sqrt{\frac{d \log n}{n}}\right)$$

**LLMへの応用：**

$d \sim N$（パラメータ数）なら

$$O\left(\sqrt{\frac{N \log n}{n}}\right)$$

LLMは $N \sim n$ なので境界が緩すぎる → 他の理論が必要

### B.10.2 PAC学習可能性

**定義：**

仮説クラス $\mathcal{H}$ がPAC学習可能とは、任意の $\epsilon, \delta > 0$ に対して、サンプル数

$$n \geq \text{poly}(1/\epsilon, 1/\delta, \text{complexity}(\mathcal{H}))$$

で、確率 $1-\delta$ 以上で誤差 $\epsilon$ 以下の仮説を出力できる。

**LLMの課題：**

複雑性が極めて高く、標準的PAC理論では扱いにくい

---

## まとめ

この付録では、LLMの理論的基盤となる数学定理を証明とともにまとめました。

**カテゴリー：**

✅ **基礎数学**: Jensen不等式、Cauchy-Schwarz  
✅ **確率論**: 大数の法則、中心極限定理  
✅ **最適化**: 凸最適化、勾配降下法  
✅ **情報理論**: Gibbs不等式、データ処理不等式  
✅ **線形代数**: SVD、低ランク近似  
✅ **ニューラルネット**: 普遍近似定理、Backpropagation  
✅ **Transformer**: Attention複雑性、位置エンコーディング  
✅ **スケーリング**: べき乗則  
✅ **強化学習**: ベルマン方程式、Policy Gradient  
✅ **汎化理論**: VC次元、PAC学習

**さらに学ぶには：**

- 確率論：測度論的確率論
- 最適化：凸解析、変分法
- 情報理論：符号理論、レート歪み理論
- 機械学習：統計的学習理論

---

**📖 前の付録：[付録A 数学記号の総まとめ](./付録A_数学記号の総まとめ.md)**  
**📖 次の付録：[付録C 数値計算の実装](./付録C_数値計算の実装.md)**
