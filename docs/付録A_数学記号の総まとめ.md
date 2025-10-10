# 付録A：数学記号の総まとめ

本ガイドブックで使用した数学記号を体系的にまとめます。

---

## A.1 集合と論理

### A.1.1 集合記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $\in$ | 属する | $x \in \mathbb{R}$ (xは実数) |
| $\notin$ | 属さない | $x \notin \mathbb{Z}$ (xは整数でない) |
| $\subset$ | 部分集合 | $\mathbb{N} \subset \mathbb{R}$ |
| $\subseteq$ | 部分集合（等しい場合含む） | $A \subseteq B$ |
| $\cup$ | 和集合 | $A \cup B$ |
| $\cap$ | 積集合 | $A \cap B$ |
| $\setminus$ | 差集合 | $A \setminus B$ |
| $\emptyset$ | 空集合 | $A \cap B = \emptyset$ |
| $\mathbb{N}$ | 自然数 | $\{0, 1, 2, 3, ...\}$ |
| $\mathbb{Z}$ | 整数 | $\{..., -1, 0, 1, ...\}$ |
| $\mathbb{Q}$ | 有理数 | $\{p/q \mid p,q \in \mathbb{Z}, q \neq 0\}$ |
| $\mathbb{R}$ | 実数 | 全ての実数 |
| $\mathbb{C}$ | 複素数 | $\{a + bi \mid a,b \in \mathbb{R}\}$ |
| $\mathbb{R}^n$ | n次元実数ベクトル空間 | $\{(x_1, ..., x_n) \mid x_i \in \mathbb{R}\}$ |
| $\|A\|$ | 集合の要素数（濃度） | $\|\{1,2,3\}\| = 3$ |

### A.1.2 論理記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $\land$ | かつ（論理積） | $P \land Q$ |
| $\lor$ | または（論理和） | $P \lor Q$ |
| $\neg$ | 否定 | $\neg P$ |
| $\Rightarrow$ | ならば（含意） | $P \Rightarrow Q$ |
| $\Leftrightarrow$ | 同値 | $P \Leftrightarrow Q$ |
| $\forall$ | 全ての（全称量化） | $\forall x \in \mathbb{R}, x^2 \geq 0$ |
| $\exists$ | 存在する（存在量化） | $\exists x \in \mathbb{R}, x^2 = 2$ |
| $\exists!$ | 一意に存在する | $\exists! x, x + 5 = 0$ |

### A.1.3 集合の記法

**集合の内包的記法：**

$$\{x \mid P(x)\}$$

「条件 $P(x)$ を満たす $x$ の集合」

**例：**

$$\{x \in \mathbb{R} \mid x^2 < 4\} = (-2, 2)$$

---

## A.2 関数と写像

### A.2.1 関数記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $f: A \to B$ | $A$ から $B$ への関数 | $f: \mathbb{R} \to \mathbb{R}$ |
| $f(x)$ | $x$ の像 | $f(2) = 4$ |
| $f^{-1}$ | 逆関数 | $f^{-1}(y)$ |
| $f \circ g$ | 合成関数 | $(f \circ g)(x) = f(g(x))$ |
| $\text{dom}(f)$ | 定義域 | $\text{dom}(\sqrt{x}) = [0, \infty)$ |
| $\text{range}(f)$ | 値域 | $\text{range}(x^2) = [0, \infty)$ |
| $\arg\max$ | 最大値を与える引数 | $\arg\max_x f(x)$ |
| $\arg\min$ | 最小値を与える引数 | $\arg\min_x f(x)$ |

### A.2.2 特殊関数

| 記号 | 意味 | 定義 |
|------|------|------|
| $\exp(x)$ | 指数関数 | $e^x$ |
| $\ln(x)$ | 自然対数 | $\log_e(x)$ |
| $\log(x)$ | 対数（底10または文脈依存） | $\log_{10}(x)$ or $\ln(x)$ |
| $\lfloor x \rfloor$ | 床関数（切り捨て） | $\lfloor 3.7 \rfloor = 3$ |
| $\lceil x \rceil$ | 天井関数（切り上げ） | $\lceil 3.2 \rceil = 4$ |
| $\text{sgn}(x)$ | 符号関数 | $1 (x>0), 0 (x=0), -1 (x<0)$ |
| $\|x\|$ | 絶対値 | $\|-5\| = 5$ |
| $\mathbb{1}_A(x)$ | 指示関数 | $1 (x \in A), 0 (x \notin A)$ |

---

## A.3 微積分

### A.3.1 微分記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $\frac{df}{dx}$ | 導関数 | $\frac{d(x^2)}{dx} = 2x$ |
| $f'(x)$ | 導関数（ラグランジュ記法） | $(x^2)' = 2x$ |
| $\dot{f}$ | 時間微分 | $\dot{x} = \frac{dx}{dt}$ |
| $\partial f/\partial x$ | 偏微分 | $\frac{\partial}{\partial x}(x^2 + y^2) = 2x$ |
| $\nabla f$ | 勾配（gradient） | $\nabla f = \left(\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n}\right)$ |
| $\nabla^2 f$ | ラプラシアン | $\sum_{i=1}^n \frac{\partial^2 f}{\partial x_i^2}$ |
| $\frac{\partial^2 f}{\partial x \partial y}$ | 2階偏微分 | 混合偏微分 |
| $H_f$ | ヘッセ行列 | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ |
| $J_f$ | ヤコビ行列 | $J_{ij} = \frac{\partial f_i}{\partial x_j}$ |

### A.3.2 積分記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $\int f(x) dx$ | 不定積分 | $\int x dx = \frac{x^2}{2} + C$ |
| $\int_a^b f(x) dx$ | 定積分 | $\int_0^1 x dx = \frac{1}{2}$ |
| $\iint f(x,y) dxdy$ | 二重積分 | 領域上の積分 |
| $\oint f(x) dx$ | 閉曲線上の積分 | 周回積分 |

### A.3.3 極限記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $\lim_{x \to a} f(x)$ | 極限 | $\lim_{x \to 0} \frac{\sin x}{x} = 1$ |
| $\lim_{x \to a^+}$ | 右極限 | $x \to a$ を正の方向から |
| $\lim_{x \to a^-}$ | 左極限 | $x \to a$ を負の方向から |
| $\lim_{n \to \infty}$ | 無限大への極限 | $\lim_{n \to \infty} \frac{1}{n} = 0$ |
| $o(g)$ | 小オー記法 | $f = o(g)$ if $\lim \frac{f}{g} = 0$ |
| $O(g)$ | 大オー記法 | $f = O(g)$ if $\limsup \frac{f}{g} < \infty$ |
| $\Theta(g)$ | シータ記法 | $f = \Theta(g)$ if $f = O(g)$ and $g = O(f)$ |
| $\Omega(g)$ | オメガ記法 | $f = \Omega(g)$ if $g = O(f)$ |

---

## A.4 線形代数

### A.4.1 ベクトル記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $\mathbf{v}$ | ベクトル | $\mathbf{v} = (v_1, v_2, v_3)$ |
| $\|\mathbf{v}\|$ | ベクトルのノルム | $\|\mathbf{v}\|_2 = \sqrt{v_1^2 + ... + v_n^2}$ |
| $\|\mathbf{v}\|_p$ | $L^p$ ノルム | $\|\mathbf{v}\|_p = \left(\sum_i \|v_i\|^p\right)^{1/p}$ |
| $\|\mathbf{v}\|_1$ | $L^1$ ノルム（マンハッタン距離） | $\sum_i \|v_i\|$ |
| $\|\mathbf{v}\|_2$ | $L^2$ ノルム（ユークリッド距離） | $\sqrt{\sum_i v_i^2}$ |
| $\|\mathbf{v}\|_\infty$ | $L^\infty$ ノルム（最大ノルム） | $\max_i \|v_i\|$ |
| $\langle \mathbf{u}, \mathbf{v} \rangle$ | 内積 | $\sum_i u_i v_i$ |
| $\mathbf{u} \cdot \mathbf{v}$ | 内積（別記法） | $\sum_i u_i v_i$ |
| $\mathbf{u} \times \mathbf{v}$ | 外積（3次元） | クロス積 |
| $\mathbf{u} \otimes \mathbf{v}$ | テンソル積 | $(\mathbf{u} \otimes \mathbf{v})_{ij} = u_i v_j$ |
| $\mathbf{0}$ | ゼロベクトル | $(0, 0, ..., 0)$ |
| $\mathbf{e}_i$ | 標準基底ベクトル | $i$ 番目が1、他は0 |

### A.4.2 行列記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $A$ | 行列 | $A \in \mathbb{R}^{m \times n}$ |
| $A^T$ | 転置行列 | $A_{ij}^T = A_{ji}$ |
| $A^{-1}$ | 逆行列 | $AA^{-1} = I$ |
| $A^*$ | 随伴行列（複素共役転置） | $A_{ij}^* = \overline{A_{ji}}$ |
| $\det(A)$ | 行列式 | スカラー値 |
| $\text{tr}(A)$ | トレース（対角和） | $\sum_i A_{ii}$ |
| $\text{rank}(A)$ | 階数 | 線形独立な行（列）の数 |
| $\text{diag}(a_1, ..., a_n)$ | 対角行列 | 対角成分が $a_i$ |
| $I$ or $I_n$ | 単位行列 | $I_{ij} = \delta_{ij}$ |
| $0$ | ゼロ行列 | 全要素が0 |
| $A \odot B$ | アダマール積（要素ごと） | $`(A \odot B)_{ij} = A_{ij} B_{ij}`$ |
| $A \otimes B$ | クロネッカー積 | テンソル積 |
| $\lambda$ | 固有値 | $A\mathbf{v} = \lambda \mathbf{v}$ |
| $\mathbf{v}$ | 固有ベクトル | $A\mathbf{v} = \lambda \mathbf{v}$ |
| $\|A\|$ | 行列ノルム | $\|A\|_2, \|A\|_F$ など |
| $\|A\|_F$ | フロベニウスノルム | $\sqrt{\sum_{ij} A_{ij}^2}$ |

---

## A.5 確率論

### A.5.1 確率記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $P(A)$ | 事象Aの確率 | $P(\text{表}) = 0.5$ |
| $P(A \mid B)$ | 条件付き確率 | $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$ |
| $P(A, B)$ | 同時確率 | $P(A \cap B)$ |
| $X \sim D$ | 確率分布Dに従う | $X \sim \mathcal{N}(0, 1)$ |
| $\mathbb{E}[X]$ | 期待値 | $\mathbb{E}[X] = \sum_x x P(X=x)$ |
| $\mathbb{E}_X[f(X)]$ | 期待値（明示） | $X$ に関する期待値 |
| $\text{Var}(X)$ | 分散 | $\mathbb{E}[(X - \mathbb{E}[X])^2]$ |
| $\text{Cov}(X, Y)$ | 共分散 | $\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$ |
| $\text{Corr}(X, Y)$ | 相関係数 | $\frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$ |
| $\sigma^2$ | 分散 | $\text{Var}(X)$ |
| $\sigma$ | 標準偏差 | $\sqrt{\text{Var}(X)}$ |

### A.5.2 確率分布

| 記号 | 意味 | パラメータ |
|------|------|-----------|
| $\mathcal{N}(\mu, \sigma^2)$ | 正規分布（ガウス分布） | 平均 $\mu$、分散 $\sigma^2$ |
| $\text{Ber}(p)$ | ベルヌーイ分布 | 成功確率 $p$ |
| $\text{Bin}(n, p)$ | 二項分布 | 試行回数 $n$、確率 $p$ |
| $\text{Cat}(\mathbf{p})$ | カテゴリカル分布 | 確率ベクトル $\mathbf{p}$ |
| $\text{Multi}(n, \mathbf{p})$ | 多項分布 | 試行回数 $n$、確率 $\mathbf{p}$ |
| $\text{Pois}(\lambda)$ | ポアソン分布 | 平均 $\lambda$ |
| $\text{Exp}(\lambda)$ | 指数分布 | レート $\lambda$ |
| $\text{Unif}(a, b)$ | 一様分布 | 範囲 $[a, b]$ |

### A.5.3 情報理論

| 記号 | 意味 | 定義 |
|------|------|------|
| $H(X)$ | エントロピー | $-\sum_x P(x) \log P(x)$ |
| $H(X \mid Y)$ | 条件付きエントロピー | $-\sum_{x,y} P(x,y) \log P(x \mid y)$ |
| $I(X; Y)$ | 相互情報量 | $H(X) - H(X \mid Y)$ |
| $D_{\text{KL}}(P \| Q)$ | KLダイバージェンス | $\sum_x P(x) \log \frac{P(x)}{Q(x)}$ |
| $D_{\text{JS}}(P \| Q)$ | JSダイバージェンス | $\frac{1}{2}D_{\text{KL}}(P \| M) + \frac{1}{2}D_{\text{KL}}(Q \| M)$ |
| $H(p, q)$ | クロスエントロピー | $-\sum_x p(x) \log q(x)$ |

---

## A.6 LLM固有の記法

### A.6.1 系列とトークン

| 記号 | 意味 | 例 |
|------|------|-----|
| $\mathcal{V}$ | 語彙（vocabulary） | 全トークンの集合 |
| $\|V\|$ | 語彙サイズ | トークン数 |
| $x_t$ | 位置 $t$ のトークン | $x_3 = \text{「猫」}$ |
| $x_{1:n}$ | トークン系列 | $(x_1, x_2, ..., x_n)$ |
| $x_{\text{<}t}$ | 位置 $t$ より前 | $(x_1, ..., x_{t-1})$ |
| $x_{\leq t}$ | 位置 $t$ まで | $(x_1, ..., x_t)$ |
| $x_{>t}$ | 位置 $t$ より後 | $(x_{t+1}, ..., x_n)$ |
| $n$ | 系列長（文脈長） | トークン数 |

### A.6.2 モデルパラメータ

| 記号 | 意味 | 例 |
|------|------|-----|
| $\theta$ | モデルパラメータ | 全ての重み |
| $W$ | 重み行列 | $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ |
| $b$ | バイアスベクトル | $b \in \mathbb{R}^{d}$ |
| $d$ | モデル次元（隠れ層サイズ） | 768, 1024, 4096など |
| $d_{\text{model}}$ | モデル次元 | $d$ と同じ |
| $d_k$ | Key/Queryの次元 | 通常 $d/h$ |
| $d_v$ | Valueの次元 | 通常 $d/h$ |
| $d_{\text{ff}}$ | FFN中間層次元 | 通常 $4d$ |
| $h$ or $H$ | ヘッド数 | Multi-Head Attention |
| $L$ | 層数 | Transformer層の数 |
| $N$ | パラメータ総数 | モデルサイズ |

### A.6.3 Attention機構

| 記号 | 意味 | 定義 |
|------|------|------|
| $Q$ | Query行列 | $Q = XW_Q$ |
| $K$ | Key行列 | $K = XW_K$ |
| $V$ | Value行列 | $V = XW_V$ |
| $\text{Attention}(Q,K,V)$ | Attention関数 | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ |
| $A$ | Attention重み | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ |
| $\text{head}_i$ | $i$ 番目のヘッド | 個別のAttention |
| $\text{MultiHead}(Q,K,V)$ | Multi-Head Attention | ヘッドの連結 |

### A.6.4 損失関数と学習

| 記号 | 意味 | 定義 |
|------|------|------|
| $\mathcal{L}$ | 損失関数 | $\mathcal{L}(\theta; \mathcal{D})$ |
| $\mathcal{L}_{\text{LM}}$ | 言語モデリング損失 | $-\sum_t \log P(x_t \mid x_{\text{<}t})$ |
| $\mathcal{L}_{\text{CE}}$ | クロスエントロピー損失 | $-\sum_i y_i \log \hat{y}_i$ |
| $\eta$ | 学習率 | Adam等のハイパーパラメータ |
| $\beta_1, \beta_2$ | モーメント係数 | Adamのハイパーパラメータ |
| $\epsilon$ | 数値安定化項 | 小さな定数（例: $10^{-8}$） |
| $\lambda$ | 正則化係数 | Weight Decay等 |
| $\nabla_\theta \mathcal{L}$ | 勾配 | 損失の微分 |

### A.6.5 評価指標

| 記号 | 意味 | 定義 |
|------|------|------|
| $\text{PPL}$ | Perplexity | $\exp(\mathcal{L}_{\text{LM}})$ |
| $\text{BLEU}$ | BLEU スコア | 機械翻訳評価 |
| $\text{ROUGE}$ | ROUGE スコア | 要約評価 |
| $F_1$ | F1スコア | $\frac{2 \cdot P \cdot R}{P + R}$ |
| $P$ | Precision（精度） | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ |
| $R$ | Recall（再現率） | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ |

---

## A.7 特殊記号と演算子

### A.7.1 総和と総積

| 記号 | 意味 | 例 |
|------|------|-----|
| $\sum_{i=1}^n$ | 総和 | $\sum_{i=1}^n i = \frac{n(n+1)}{2}$ |
| $\prod_{i=1}^n$ | 総積 | $\prod_{i=1}^n i = n!$ |
| $\sum_{i \in S}$ | 集合上の総和 | $S$ の要素についての和 |
| $\bigoplus$ | 排他的論理和（XOR） | ビット演算 |

### A.7.2 その他の数学記号

| 記号 | 意味 | 例 |
|------|------|-----|
| $\approx$ | 近似 | $\pi \approx 3.14$ |
| $\propto$ | 比例 | $y \propto x$ |
| $\equiv$ | 恒等的に等しい | $\sin^2 x + \cos^2 x \equiv 1$ |
| $:=$ | 定義 | $f(x) := x^2$ |
| $\triangleq$ | 定義（別記法） | $a \triangleq b$ |
| $\perp$ | 独立 | $X \perp Y$ |
| $\parallel$ | 平行 | ベクトルの並列性 |
| $\perp\!\!\!\perp$ | 条件付き独立 | $X \perp\!\!\!\perp Y \mid Z$ |
| $\ll$ | 遥かに小さい | $x \ll y$ |
| $\gg$ | 遥かに大きい | $x \gg y$ |
| $\sim$ | 同分布 | $X \sim Y$ |
| $\asymp$ | 漸近的に等しい | $f \asymp g$ |

### A.7.3 ギリシャ文字

**小文字：**

| 文字 | 読み | よく使う意味 |
|------|------|-------------|
| $\alpha$ | アルファ | 学習率、スケーリング指数 |
| $\beta$ | ベータ | モーメント係数 |
| $\gamma$ | ガンマ | 割引率（強化学習） |
| $\delta$ | デルタ | 小さな変化 |
| $\epsilon$ | イプシロン | 小さな定数 |
| $\zeta$ | ゼータ | |
| $\eta$ | イータ | 学習率 |
| $\theta$ | シータ | パラメータ |
| $\lambda$ | ラムダ | 正則化係数、固有値 |
| $\mu$ | ミュー | 平均 |
| $\nu$ | ニュー | |
| $\xi$ | クサイ | |
| $\pi$ | パイ | 円周率、方策（強化学習） |
| $\rho$ | ロー | 相関係数 |
| $\sigma$ | シグマ | 標準偏差 |
| $\tau$ | タウ | 時定数 |
| $\phi$ | ファイ | 特徴関数 |
| $\chi$ | カイ | |
| $\psi$ | プサイ | |
| $\omega$ | オメガ | 角周波数 |

**大文字：**

| 文字 | 読み | よく使う意味 |
|------|------|-------------|
| $\Gamma$ | ガンマ | ガンマ関数 |
| $\Delta$ | デルタ | 差分 |
| $\Theta$ | シータ | 計算量（大オー記法） |
| $\Lambda$ | ラムダ | |
| $\Sigma$ | シグマ | 総和、共分散行列 |
| $\Phi$ | ファイ | |
| $\Psi$ | プサイ | |
| $\Omega$ | オメガ | 計算量（下界） |

---

## A.8 活性化関数

| 関数 | 定義 | 導関数 |
|------|------|--------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ |
| ReLU | $\text{ReLU}(x) = \max(0, x)$ | $`\begin{cases}1 & x>0\\0 & x\leq 0\end{cases}`$ |
| GELU | $\text{GELU}(x) = x\Phi(x)$ | 複雑（数値計算） |
| Softmax | $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | $\text{softmax}(x_i)(1-\text{softmax}(x_i))$ |

---

## A.9 索引記法の慣例

### A.9.1 添字の意味

本ガイドブックでの慣例：

| 添字 | 意味 | 例 |
|------|------|-----|
| $i, j, k$ | インデックス（一般） | $x_i, A_{ij}$ |
| $t$ | 時間ステップ、系列位置 | $x_t$ |
| $n$ | 系列長、データ数 | $x_{1:n}$ |
| $d$ | 次元 | $\mathbb{R}^d$ |
| $l$ | 層番号 | $h^{(l)}$ |
| $m$ | バッチサイズ | $m$ 個のサンプル |
| $b$ | バッチインデックス | $x^{(b)}$ |

### A.9.2 括弧の使い分け

| 括弧 | 用途 | 例 |
|------|------|-----|
| $(\ )$ | 関数の引数、順序組 | $f(x)$, $(a, b)$ |
| $[\ ]$ | 配列、行列、区間 | $A[i, j]$, $[0, 1]$ |
| $\{\ \}$ | 集合 | $\{1, 2, 3\}$ |
| $\langle\ \rangle$ | 内積、期待値 | $\langle x, y \rangle$ |
| $\|\ \|$ | ノルム、絶対値、集合の濃度 | $\|x\|$, $\|A\|$ |

---

## A.10 略語と用語

### A.10.1 一般的な略語

| 略語 | 完全形 | 意味 |
|------|--------|------|
| AI | Artificial Intelligence | 人工知能 |
| ML | Machine Learning | 機械学習 |
| DL | Deep Learning | 深層学習 |
| NLP | Natural Language Processing | 自然言語処理 |
| CV | Computer Vision | コンピュータビジョン |
| RL | Reinforcement Learning | 強化学習 |

### A.10.2 LLM関連

| 略語 | 完全形 | 意味 |
|------|--------|------|
| LLM | Large Language Model | 大規模言語モデル |
| GPT | Generative Pre-trained Transformer | |
| BERT | Bidirectional Encoder Representations from Transformers | |
| T5 | Text-to-Text Transfer Transformer | |
| PPO | Proximal Policy Optimization | |
| RLHF | Reinforcement Learning from Human Feedback | |
| CoT | Chain-of-Thought | |
| ICL | In-Context Learning | |
| FFN | Feed-Forward Network | |
| MHA | Multi-Head Attention | |
| MLP | Multi-Layer Perceptron | |

### A.10.3 訓練・評価

| 略語 | 完全形 | 意味 |
|------|--------|------|
| SGD | Stochastic Gradient Descent | 確率的勾配降下法 |
| Adam | Adaptive Moment Estimation | |
| PPL | Perplexity | パープレキシティ |
| BLEU | Bilingual Evaluation Understudy | |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation | |
| F1 | F1 Score | |
| MLE | Maximum Likelihood Estimation | 最尤推定 |
| MAP | Maximum A Posteriori | 最大事後確率 |

---

## A.11 クイックリファレンス

### よく使う公式

**基本統計：**

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**ベイズの定理：**

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

**勾配：**

$$\nabla_x f = \left(\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n}\right)$$

**連鎖律：**

$$\frac{df}{dx} = \frac{df}{dy} \cdot \frac{dy}{dx}$$

**KLダイバージェンス：**

$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} \geq 0$$

**Attention：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Softmax：**

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**クロスエントロピー：**

$$H(p, q) = -\sum_x p(x) \log q(x)$$

---

## まとめ

この付録では、本ガイドブック全体で使用した数学記号を体系的にまとめました。

**使い方：**
- 本文で不明な記号に出会ったら参照
- 数式を書く際の標準記法として利用
- 他の文献を読む際の橋渡しとして活用

**次の付録へ：**
- **付録B**: 重要定理と証明
- **付録C**: 数値計算の実装
- **付録D**: ベンチマークデータセット
- **付録E**: 参考文献

---

**📖 本文へ戻る：[目次](../README.md)**  
**📖 次の付録：[付録B 重要定理と証明](./付録B_重要定理と証明.md)**
