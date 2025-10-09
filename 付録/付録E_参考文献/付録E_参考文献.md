# 付録E：参考文献

LLM理論を学ぶための重要な論文、書籍、リソースをまとめます。

---

## E.1 基礎論文（必読）

### E.1.1 Transformer の誕生

**Attention Is All You Need (2017)**
- 著者: Vaswani et al. (Google)
- 会議: NeurIPS 2017
- リンク: https://arxiv.org/abs/1706.03762
- 重要性: ⭐⭐⭐⭐⭐
- 内容: Self-Attention機構、Multi-Head Attention、位置エンコーディング

**重要な貢献:**
```
✓ RNNを使わない系列モデル
✓ Scaled Dot-Product Attention
✓ 並列化可能なアーキテクチャ
✓ 翻訳タスクでSOTA達成
```

### E.1.2 GPTシリーズ

**Improving Language Understanding by Generative Pre-Training (GPT-1, 2018)**
- 著者: Radford et al. (OpenAI)
- リンク: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 事前学習 + ファインチューニングのパラダイム

**Language Models are Unsupervised Multitask Learners (GPT-2, 2019)**
- 著者: Radford et al. (OpenAI)
- リンク: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- 重要性: ⭐⭐⭐⭐⭐
- 内容: Zero-shot学習、スケールアップ（1.5B パラメータ）

**Language Models are Few-Shot Learners (GPT-3, 2020)**
- 著者: Brown et al. (OpenAI)
- 会議: NeurIPS 2020
- リンク: https://arxiv.org/abs/2005.14165
- 重要性: ⭐⭐⭐⭐⭐
- 内容: In-Context Learning、Few-shot学習、創発的能力（175B パラメータ）

### E.1.3 BERT と双方向モデル

**BERT: Pre-training of Deep Bidirectional Transformers (2018)**
- 著者: Devlin et al. (Google)
- 会議: NAACL 2019
- リンク: https://arxiv.org/abs/1810.04805
- 重要性: ⭐⭐⭐⭐⭐
- 内容: Masked Language Modeling、双方向エンコーダー

**RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)**
- 著者: Liu et al. (Facebook)
- リンク: https://arxiv.org/abs/1907.11692
- 重要性: ⭐⭐⭐⭐
- 内容: BERTの訓練改善、NSP除去、動的マスキング

---

## E.2 スケーリング則

**Scaling Laws for Neural Language Models (2020)**
- 著者: Kaplan et al. (OpenAI)
- リンク: https://arxiv.org/abs/2001.08361
- 重要性: ⭐⭐⭐⭐⭐
- 内容: べき乗則 $L \propto N^{-\alpha}$、最適配分

**Training Compute-Optimal Large Language Models (Chinchilla, 2022)**
- 著者: Hoffmann et al. (DeepMind)
- リンク: https://arxiv.org/abs/2203.15556
- 重要性: ⭐⭐⭐⭐⭐
- 内容: データとモデルサイズの同時スケーリング、Chinchilla最適性

**Emergent Abilities of Large Language Models (2022)**
- 著者: Wei et al. (Google)
- リンク: https://arxiv.org/abs/2206.07682
- 重要性: ⭐⭐⭐⭐
- 内容: 創発的能力の分析、スケールによる質的変化

---

## E.3 学習アルゴリズム

### E.3.1 最適化

**Adam: A Method for Stochastic Optimization (2014)**
- 著者: Kingma & Ba
- 会議: ICLR 2015
- リンク: https://arxiv.org/abs/1412.6980
- 重要性: ⭐⭐⭐⭐⭐
- 内容: Adaptive learning rate、モーメント推定

**Decoupled Weight Decay Regularization (AdamW, 2017)**
- 著者: Loshchilov & Hutter
- 会議: ICLR 2019
- リンク: https://arxiv.org/abs/1711.05101
- 重要性: ⭐⭐⭐⭐
- 内容: Weight DecayとL2正則化の分離

### E.3.2 強化学習とRLHF

**Proximal Policy Optimization Algorithms (PPO, 2017)**
- 著者: Schulman et al. (OpenAI)
- リンク: https://arxiv.org/abs/1707.06347
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 方策最適化、クリッピング目的関数

**Training language models to follow instructions with human feedback (InstructGPT, 2022)**
- 著者: Ouyang et al. (OpenAI)
- リンク: https://arxiv.org/abs/2203.02155
- 重要性: ⭐⭐⭐⭐⭐
- 内容: RLHF、報酬モデリング、人間の嗜好との整合

**Constitutional AI: Harmlessness from AI Feedback (2022)**
- 著者: Bai et al. (Anthropic)
- リンク: https://arxiv.org/abs/2212.08073
- 重要性: ⭐⭐⭐⭐
- 内容: AI自身によるフィードバック、憲法的原則

---

## E.4 プロンプトエンジニアリング

**Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2022)**
- 著者: Wei et al. (Google)
- リンク: https://arxiv.org/abs/2201.11903
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 思考連鎖プロンプティング、推論能力の向上

**Self-Consistency Improves Chain of Thought Reasoning (2022)**
- 著者: Wang et al. (Google)
- リンク: https://arxiv.org/abs/2203.11171
- 重要性: ⭐⭐⭐⭐
- 内容: 複数サンプリング + 多数決

**Large Language Models are Zero-Shot Reasoners (2022)**
- 著者: Kojima et al.
- リンク: https://arxiv.org/abs/2205.11916
- 重要性: ⭐⭐⭐⭐
- 内容: "Let's think step by step" の効果

---

## E.5 効率化技術

### E.5.1 量子化とプルーニング

**LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)**
- 著者: Dettmers et al.
- リンク: https://arxiv.org/abs/2208.07339
- 重要性: ⭐⭐⭐⭐
- 内容: 8bit量子化、外れ値の扱い

**GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (2022)**
- 著者: Frantar et al.
- リンク: https://arxiv.org/abs/2210.17323
- 重要性: ⭐⭐⭐⭐
- 内容: Post-training量子化、精度保持

### E.5.2 Parameter-Efficient Fine-Tuning

**LoRA: Low-Rank Adaptation of Large Language Models (2021)**
- 著者: Hu et al. (Microsoft)
- 会議: ICLR 2022
- リンク: https://arxiv.org/abs/2106.09685
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 低ランク適応、パラメータ効率的ファインチューニング

**Prefix-Tuning: Optimizing Continuous Prompts for Generation (2021)**
- 著者: Li & Liang (Stanford)
- 会議: ACL 2021
- リンク: https://arxiv.org/abs/2101.00190
- 重要性: ⭐⭐⭐⭐
- 内容: 連続的プロンプト学習

### E.5.3 効率的Attention

**FlashAttention: Fast and Memory-Efficient Exact Attention (2022)**
- 著者: Dao et al. (Stanford)
- リンク: https://arxiv.org/abs/2205.14135
- 重要性: ⭐⭐⭐⭐⭐
- 内容: メモリ階層最適化、IO効率化

**Longformer: The Long-Document Transformer (2020)**
- 著者: Beltagy et al. (Allen AI)
- リンク: https://arxiv.org/abs/2004.05150
- 重要性: ⭐⭐⭐⭐
- 内容: Sparse Attention、長文脈処理

---

## E.6 多言語・多モーダル

**Unsupervised Cross-lingual Representation Learning at Scale (XLM-R, 2019)**
- 著者: Conneau et al. (Facebook)
- 会議: ACL 2020
- リンク: https://arxiv.org/abs/1911.02116
- 重要性: ⭐⭐⭐⭐
- 内容: 100言語の多言語モデル

**Learning Transferable Visual Models From Natural Language Supervision (CLIP, 2021)**
- 著者: Radford et al. (OpenAI)
- 会議: ICML 2021
- リンク: https://arxiv.org/abs/2103.00020
- 重要性: ⭐⭐⭐⭐⭐
- 内容: Vision-Language事前学習、コントラスト学習

**Flamingo: a Visual Language Model for Few-Shot Learning (2022)**
- 著者: Alayrac et al. (DeepMind)
- リンク: https://arxiv.org/abs/2204.14198
- 重要性: ⭐⭐⭐⭐
- 内容: 視覚言語モデル、Few-shot学習

---

## E.7 解釈可能性と安全性

**A Mathematical Framework for Transformer Circuits (2021)**
- 著者: Elhage et al. (Anthropic)
- リンク: https://transformer-circuits.pub/2021/framework/index.html
- 重要性: ⭐⭐⭐⭐
- 内容: Induction Heads、メカニスティック解釈可能性

**In-context Learning and Induction Heads (2022)**
- 著者: Olsson et al. (Anthropic)
- リンク: https://arxiv.org/abs/2209.11895
- 重要性: ⭐⭐⭐⭐
- 内容: ICLのメカニズム、Induction Heads

**Red Teaming Language Models to Reduce Harms (2022)**
- 著者: Perez et al. (Anthropic)
- リンク: https://arxiv.org/abs/2209.07858
- 重要性: ⭐⭐⭐⭐
- 内容: レッドチーミング、有害性削減

---

## E.8 理論的基礎

**Understanding Deep Learning Requires Rethinking Generalization (2016)**
- 著者: Zhang et al.
- 会議: ICLR 2017
- リンク: https://arxiv.org/abs/1611.03530
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 過剰パラメータ化、汎化の謎

**Deep Double Descent (2019)**
- 著者: Nakkiran et al.
- リンク: https://arxiv.org/abs/1912.02292
- 重要性: ⭐⭐⭐⭐
- 内容: Double Descent現象、汎化曲線

**Grokking: Generalization Beyond Overfitting (2021)**
- 著者: Power et al. (OpenAI)
- リンク: https://arxiv.org/abs/2201.02177
- 重要性: ⭐⭐⭐
- 内容: 遅延汎化、Grokking現象

---

## E.9 書籍

### E.9.1 深層学習の基礎

**Deep Learning (2016)**
- 著者: Ian Goodfellow, Yoshua Bengio, Aaron Courville
- 出版社: MIT Press
- リンク: https://www.deeplearningbook.org/
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 深層学習の包括的教科書

**Neural Networks and Deep Learning (2018)**
- 著者: Charu C. Aggarwal
- 出版社: Springer
- 重要性: ⭐⭐⭐⭐
- 内容: 理論と実装のバランス

### E.9.2 自然言語処理

**Speech and Language Processing (3rd ed. draft)**
- 著者: Dan Jurafsky, James H. Martin
- リンク: https://web.stanford.edu/~jurafsky/slp3/
- 重要性: ⭐⭐⭐⭐⭐
- 内容: NLPの古典的教科書、LLM章を含む

**Natural Language Processing with Transformers (2022)**
- 著者: Lewis Tunstall, Leandro von Werra, Thomas Wolf
- 出版社: O'Reilly
- 重要性: ⭐⭐⭐⭐
- 内容: HuggingFace Transformers ライブラリの実践

### E.9.3 数学的基礎

**Pattern Recognition and Machine Learning (2006)**
- 著者: Christopher M. Bishop
- 出版社: Springer
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 機械学習の数学的基礎

**Information Theory, Inference, and Learning Algorithms (2003)**
- 著者: David J.C. MacKay
- リンク: http://www.inference.org.uk/mackay/itila/
- 重要性: ⭐⭐⭐⭐
- 内容: 情報理論と機械学習

---

## E.10 オンラインリソース

### E.10.1 コース

**Stanford CS224N: Natural Language Processing with Deep Learning**
- リンク: http://web.stanford.edu/class/cs224n/
- 講師: Christopher Manning
- 重要性: ⭐⭐⭐⭐⭐
- 内容: NLP + 深層学習の体系的コース

**Stanford CS25: Transformers United**
- リンク: https://web.stanford.edu/class/cs25/
- 重要性: ⭐⭐⭐⭐
- 内容: Transformer特化コース

**Fast.ai Practical Deep Learning for Coders**
- リンク: https://course.fast.ai/
- 重要性: ⭐⭐⭐⭐
- 内容: 実践重視の深層学習

### E.10.2 ブログとチュートリアル

**The Illustrated Transformer**
- 著者: Jay Alammar
- リンク: http://jalammar.github.io/illustrated-transformer/
- 重要性: ⭐⭐⭐⭐⭐
- 内容: Transformerの視覚的解説

**The Annotated Transformer**
- 著者: Harvard NLP
- リンク: http://nlp.seas.harvard.edu/2018/04/03/attention.html
- 重要性: ⭐⭐⭐⭐⭐
- 内容: コード付き詳細解説

**Andrej Karpathy's Blog**
- リンク: http://karpathy.github.io/
- 重要性: ⭐⭐⭐⭐
- 内容: 深層学習の直感的解説

**Lil'Log (Lilian Weng's Blog)**
- リンク: https://lilianweng.github.io/
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 最新研究の丁寧な解説

### E.10.3 実装ライブラリ

**HuggingFace Transformers**
- リンク: https://github.com/huggingface/transformers
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 最も広く使われるTransformerライブラリ

**PyTorch**
- リンク: https://pytorch.org/
- 重要性: ⭐⭐⭐⭐⭐
- 内容: 深層学習フレームワーク

**JAX**
- リンク: https://github.com/google/jax
- 重要性: ⭐⭐⭐⭐
- 内容: 高性能数値計算ライブラリ

---

## E.11 会議とジャーナル

### E.11.1 主要会議

**トップティア：**
```
NeurIPS (Neural Information Processing Systems)
ICML (International Conference on Machine Learning)
ICLR (International Conference on Learning Representations)
ACL (Association for Computational Linguistics)
EMNLP (Empirical Methods in NLP)
NAACL (North American Chapter of ACL)
```

**コンピュータビジョン（マルチモーダル関連）：**
```
CVPR (Computer Vision and Pattern Recognition)
ICCV (International Conference on Computer Vision)
ECCV (European Conference on Computer Vision)
```

### E.11.2 ジャーナル

```
JMLR (Journal of Machine Learning Research)
TACL (Transactions of the ACL)
Nature Machine Intelligence
Science Robotics
```

### E.11.3 プレプリントサーバー

**arXiv**
- リンク: https://arxiv.org/
- カテゴリ: cs.CL (Computation and Language), cs.LG (Machine Learning)
- 最新論文が最も早く公開される

---

## E.12 読み方のガイド

### E.12.1 初心者向け読書順序

**Phase 1: 基礎固め**
1. Deep Learning (Goodfellow et al.) - 第1-5章
2. The Illustrated Transformer (Jay Alammar)
3. Attention Is All You Need (論文)

**Phase 2: LLMの理解**
4. BERT 論文
5. GPT-2 論文
6. GPT-3 論文
7. Scaling Laws 論文

**Phase 3: 実践と応用**
8. Natural Language Processing with Transformers (書籍)
9. HuggingFace チュートリアル
10. Chain-of-Thought 論文

**Phase 4: 最新トピック**
11. InstructGPT / RLHF 論文
12. FlashAttention 論文
13. 興味のある特定トピックの深掘り

### E.12.2 論文の読み方

**効率的な読み方：**

```
1st pass (5-10分):
  ✓ タイトル、アブストラクト
  ✓ 図表を全て見る
  ✓ 結論を読む
  → 読む価値があるか判断

2nd pass (30-60分):
  ✓ イントロダクション
  ✓ 関連研究
  ✓ 手法のハイライト
  ✓ 実験結果
  → 主要なアイデアを理解

3rd pass (数時間):
  ✓ 詳細な手法
  ✓ 証明や導出
  ✓ コードを読む/実装する
  → 完全な理解と再現
```

### E.12.3 最新情報の追い方

**日常的にチェック：**
```
✓ arXiv cs.CL / cs.LG（毎日）
✓ Twitter（研究者をフォロー）
✓ Papers with Code（実装付き論文）
✓ HuggingFace Daily Papers
```

**週次：**
```
✓ 主要研究者のブログ
✓ Reddit r/MachineLearning
✓ 研究室のニュースレター
```

**会議シーズン：**
```
✓ NeurIPS, ICML, ICLR, ACL等の採択論文リスト
✓ ワークショップとチュートリアル
✓ ベストペーパー賞
```

---

## まとめ

### 学習リソースの階層

```
入門レベル:
  📘 The Illustrated Transformer
  📘 Stanford CS224N
  📘 Fast.ai コース

中級レベル:
  📕 Deep Learning (Goodfellow et al.)
  📕 主要論文（BERT, GPT, Transformer）
  📕 HuggingFace Transformers

上級レベル:
  📗 最新論文（arXiv）
  📗 理論的基礎（汎化理論、最適化）
  📗 専門分野の深掘り

研究レベル:
  📙 未解決問題
  📙 会議発表
  📙 独自の研究
```

### 重要な教訓

**理論と実践の両輪：**
- 論文を読むだけでなく、実装する
- 実装するだけでなく、理論を理解する

**継続的学習：**
- 分野は急速に進化
- 定期的なキャッチアップが必須

**コミュニティ参加：**
- GitHub、論文ディスカッション
- 勉強会、読書会
- 自分の理解を共有

---

**このガイドブックの旅はここで終わりますが、LLMの学習は続きます。**

**Happy Learning! 🚀📚**

---

**📖 前の付録：[付録D ベンチマークデータセット](../付録D_ベンチマークデータセット/付録D_ベンチマークデータセット.md)**  
**📖 目次に戻る：[README](../../README.md)**
