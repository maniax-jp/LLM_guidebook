# 付録D：ベンチマークデータセット

LLMの評価に使用される主要なベンチマークデータセットをまとめます。

---

## D.1 言語モデリング

### D.1.1 Penn Treebank (PTB)

**概要：**
- **タスク**: 言語モデリング
- **サイズ**: 約100万語
- **特徴**: 新聞記事、文法的にアノテーション済み
- **評価指標**: Perplexity (PPL)

**データ構成：**
```
Train: 42,068文
Valid: 3,370文
Test: 3,761文
語彙サイズ: 10,000
```

**入手先：**
- https://catalog.ldc.upenn.edu/LDC99T42
- Tomas Mikolov's preprocessed version (よく使用される)

**ベースライン：**
```
LSTM (2017): PPL ~60
Transformer (2018): PPL ~50
GPT-2 Small: PPL ~35
```

### D.1.2 WikiText-103

**概要：**
- **タスク**: 長距離依存の言語モデリング
- **サイズ**: 約103M語
- **特徴**: Wikipedia記事、PTBより大規模で現代的
- **評価指標**: Perplexity

**データ構成：**
```
Train: 28,475記事
Valid: 60記事
Test: 60記事
語彙サイズ: ~268,000
平均記事長: ~3,600語
```

**入手先：**
- https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

**ベースライン：**
```
LSTM (2018): PPL ~40
Transformer-XL: PPL ~18.3
GPT-2 Medium: PPL ~15
```

### D.1.3 The Pile

**概要：**
- **タスク**: 多様なドメインの言語モデリング
- **サイズ**: 825GB（未圧縮）
- **特徴**: 22の多様なデータソース
- **評価指標**: Per-source Perplexity

**データソース（一部）：**
```
- PubMed Central (科学論文)
- ArXiv (物理・数学論文)
- GitHub (コード)
- StackExchange (Q&A)
- Wikipedia
- Books3
- OpenWebText2
など22ソース
```

**入手先：**
- https://pile.eleuther.ai/

**使用例：**
- GPT-Neo, GPT-J の訓練データ

---

## D.2 理解タスク

### D.2.1 GLUE (General Language Understanding Evaluation)

**概要：**
- **タスク**: 9つの自然言語理解タスク
- **評価指標**: タスクごとの指標（Accuracy, F1等）+ 平均スコア

**タスク一覧：**

| タスク | 内容 | サイズ | 指標 |
|--------|------|--------|------|
| CoLA | 文法性判定 | 10.6K | Matthews相関係数 |
| SST-2 | 感情分析（2値） | 70K | Accuracy |
| MRPC | パラフレーズ検出 | 5.8K | F1, Accuracy |
| QQP | 質問の重複検出 | 400K | F1, Accuracy |
| STS-B | 意味的類似度 | 7K | Spearman相関 |
| MNLI | 自然言語推論 | 433K | Accuracy |
| QNLI | 質問応答NLI | 130K | Accuracy |
| RTE | 含意認識 | 2.7K | Accuracy |
| WNLI | Winograd NLI | 740 | Accuracy |

**入手先：**
- https://gluebenchmark.com/

**ベースライン（GLUE Score）：**
```
BERT-Base: 78.3
BERT-Large: 80.5
RoBERTa-Large: 88.5
GPT-3.5: ~89 (few-shot)
人間: ~87
```

### D.2.2 SuperGLUE

**概要：**
- GLUEより難しいタスク集
- **タスク数**: 8タスク
- **特徴**: より高度な推論が必要

**タスク一覧：**

| タスク | 内容 | 特徴 |
|--------|------|------|
| BoolQ | Yes/No質問応答 | Wikipedia段落ベース |
| CB | 含意/矛盾/中立の判定 | 3値分類 |
| COPA | 因果推論 | 原因・結果の選択 |
| MultiRC | 多選択肢読解 | 複数の正解 |
| ReCoRD | 穴埋め読解 | エンティティ抽出 |
| RTE | 含意認識 | GLUEより難 |
| WiC | 単語の意味 | 文脈依存の同義性 |
| WSC | Winograd Schema | 照応解決 |

**入手先：**
- https://super.gluebenchmark.com/

**ベースライン：**
```
BERT-Large: 69.0
RoBERTa-Large: 84.6
T5-11B: 89.3
GPT-3 (few-shot): 71.8
人間: 89.8
```

### D.2.3 SQuAD (Stanford Question Answering Dataset)

**概要：**
- **タスク**: 抽出型質問応答
- **サイズ**: 10万+ 質問

**バージョン：**

**SQuAD 1.1:**
```
特徴: 全ての質問に答えが存在
Train: 87,599 Q&A
Dev: 10,570 Q&A
評価指標: Exact Match (EM), F1
```

**SQuAD 2.0:**
```
特徴: 答えのない質問を含む
Train: 130,319 Q&A (50%が答えなし)
Dev: 11,873 Q&A
評価指標: EM, F1（答えなしの検出含む）
```

**入手先：**
- https://rajpurkar.github.io/SQuAD-explorer/

**ベースライン（SQuAD 2.0）：**
```
BERT-Large: EM 80.0, F1 83.1
RoBERTa-Large: EM 86.5, F1 89.4
人間: EM 86.8, F1 89.5
```

---

## D.3 生成タスク

### D.3.1 CNN/DailyMail

**概要：**
- **タスク**: 要約
- **サイズ**: 約300K記事
- **特徴**: ニュース記事 + 箇条書き要約

**データ構成：**
```
Train: 287,227
Valid: 13,368
Test: 11,490
平均記事長: ~800語
平均要約長: ~60語
```

**評価指標：**
- ROUGE-1, ROUGE-2, ROUGE-L

**入手先：**
- https://github.com/abisee/cnn-dailymail

**ベースライン（ROUGE-L）：**
```
Lead-3 baseline: 40.4
BERT+Transformer: 43.0
PEGASUS: 44.2
GPT-3.5 (zero-shot): ~42
```

### D.3.2 WMT (Workshop on Machine Translation)

**概要：**
- **タスク**: 機械翻訳
- **言語ペア**: 多数（英独、英仏、英中など）
- **評価指標**: BLEU, chrF, TER

**例：WMT 2014 英→独**
```
Train: 4.5M文ペア
Newstest2014 (test): 3,003文
```

**入手先：**
- http://www.statmt.org/wmt14/

**ベースライン（BLEU）：**
```
Transformer Base: 27.3
Transformer Big: 28.4
GPT-3 (few-shot): ~26
人間翻訳者: ~30-40（参考値）
```

### D.3.3 XSum

**概要：**
- **タスク**: 極端な要約（1文要約）
- **サイズ**: 約227K BBC記事
- **特徴**: 記事を1文に要約

**データ構成：**
```
Train: 204,045
Valid: 11,332
Test: 11,334
平均記事長: ~430語
要約長: 1文（~20語）
```

**評価指標：**
- ROUGE-1, ROUGE-2, ROUGE-L

**入手先：**
- https://github.com/EdinburghNLP/XSum

**ベースライン（ROUGE-L）：**
```
Lead-1 baseline: 16.3
PEGASUS: 23.0
BART-Large: 22.4
```

---

## D.4 推論タスク

### D.4.1 HellaSwag

**概要：**
- **タスク**: 常識推論（文の続きを選択）
- **サイズ**: 70K問題
- **特徴**: 日常シナリオの続きを4択から選択

**例：**
```
Context: "A man is sitting in a chair. He..."
A) is peeling a potato.
B) is holding a cat.
C) flies into the air.
D) starts to laugh.

正解: B（文脈依存）
```

**入手先：**
- https://rowanzellers.com/hellaswag/

**ベースライン（Accuracy）：**
```
BERT-Large: 62.3%
RoBERTa-Large: 85.6%
GPT-3: 78.9% (zero-shot)
GPT-4: 95.3%
人間: 95.6%
```

### D.4.2 PIQA (Physical Interaction QA)

**概要：**
- **タスク**: 物理的常識推論
- **サイズ**: 約21K問題
- **特徴**: 日常の物理的タスクの達成方法を2択から選択

**例：**
```
Goal: "クッキーを焼く"
A) オーブンを350度に予熱し、生地を天板に置く
B) 冷蔵庫に生地を入れて冷やす

正解: A
```

**入手先：**
- https://yonatanbisk.com/piqa/

**ベースライン（Accuracy）：**
```
RoBERTa-Large: 79.4%
GPT-3: 81.0% (zero-shot)
GPT-4: 86.2%
人間: 94.9%
```

### D.4.3 ARC (AI2 Reasoning Challenge)

**概要：**
- **タスク**: 科学的推論
- **サイズ**: 7,787問題
- **特徴**: 小学校レベルの科学試験問題

**サブセット：**

**ARC-Easy:**
```
サイズ: 5,197問題
難易度: 比較的簡単
```

**ARC-Challenge:**
```
サイズ: 2,590問題
難易度: 高（検索ベース手法で解けない）
```

**入手先：**
- https://allenai.org/data/arc

**ベースライン（ARC-Challenge Accuracy）：**
```
BERT-Large: 42.1%
RoBERTa-Large: 61.0%
GPT-3: 51.4% (zero-shot)
GPT-4: 96.3%
```

---

## D.5 多言語・多タスク

### D.5.1 XTREME

**概要：**
- **タスク**: 多言語理解・生成の総合ベンチマーク
- **言語数**: 40言語
- **タスク数**: 9タスク

**タスク一覧：**
```
1. 構文解析 (UD)
2. 固有表現認識 (PAN-X, WikiANN)
3. 質問応答 (XQuAD, MLQA, TyDiQA)
4. 文分類 (XNLI)
5. 構造化予測 (PAWS-X)
6. 文検索 (Tatoeba, BUCC)
```

**入手先：**
- https://github.com/google-research/xtreme

**ベースライン（平均スコア）：**
```
mBERT: 56.3
XLM-R Large: 74.2
mT5-XXL: 81.1
```

### D.5.2 MMLU (Massive Multitask Language Understanding)

**概要：**
- **タスク**: 57科目の多肢選択試験
- **サイズ**: 15,908問題
- **特徴**: STEM、人文、社会科学など広範囲

**科目例：**
```
- 抽象代数学
- 解剖学
- 天文学
- 経営学
- 化学
- コンピュータサイエンス
- 経済学
- 哲学
- 物理学
...（計57科目）
```

**入手先：**
- https://github.com/hendrycks/test

**ベースライン（5-shot Accuracy）：**
```
GPT-3 175B: 43.9%
GPT-3.5: 70.0%
GPT-4: 86.4%
ランダム: 25%
人間専門家: 89.8%
```

### D.5.3 BIG-bench

**概要：**
- **タスク**: 200以上の多様なタスク
- **特徴**: LLMの能力と限界を探索
- **評価**: タスクごとの多様な指標

**タスクカテゴリー：**
```
- 言語理解
- 常識推論
- 数学的推論
- コード理解
- マルチステップ推論
- 創造性
- 社会的バイアス
...
```

**入手先：**
- https://github.com/google/BIG-bench

**結果例（一部タスク）：**
```
              GPT-3  PaLM  GPT-4
数学:         30%   56%   92%
コード理解:   25%   45%   81%
常識推論:     65%   78%   91%
```

---

## D.6 特殊タスク

### D.6.1 TruthfulQA

**概要：**
- **タスク**: 事実性評価
- **サイズ**: 817問題
- **特徴**: 誤情報を含む選択肢

**カテゴリー：**
```
- 健康
- 法律
- 陰謀論
- 誤解されやすい事実
- フィクション vs 現実
- ステレオタイプ
```

**評価指標：**
```
- Truthful: 真実の回答率
- Informative: 情報量
- Both: 両方を満たす率
```

**入手先：**
- https://github.com/sylinrl/TruthfulQA

**ベースライン（Truthful+Informative）：**
```
GPT-3 175B: 29%
GPT-3.5: 47%
GPT-4: 59%
人間: 94%
```

### D.6.2 HumanEval

**概要：**
- **タスク**: コード生成
- **サイズ**: 164問題
- **特徴**: 関数の実装（Python）

**評価方法：**
- Pass@k: k個の生成候補のうち1個以上が全テストケースを通過

**例：**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    リスト内の2つの数値がthreshold未満の距離にあるか判定
    
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # LLMが実装を生成
```

**入手先:**
- https://github.com/openai/human-eval

**ベースライン（Pass@1）：**
```
GPT-3 175B: 0%
Codex 12B: 28.8%
GPT-3.5: 48.1%
GPT-4: 67.0%
```

### D.6.3 GSM8K

**概要：**
- **タスク**: 小学校レベルの数学文章題
- **サイズ**: 8,500問題
- **特徴**: 多段階推論が必要

**例：**
```
問題: 「ジョンは毎日3個のリンゴを食べます。
      1週間でリンゴは何個必要ですか？」

解答: 3個/日 × 7日 = 21個
```

**評価：**
- 最終的な数値の正確性

**入手先：**
- https://github.com/openai/grade-school-math

**ベースライン（Accuracy）：**
```
GPT-3 175B: 34.4% (few-shot)
GPT-3 + CoT: 58.1%
GPT-3.5: 57.1%
GPT-4: 92.0%
GPT-4 + CoT: 97.2%
```

---

## D.7 安全性・バイアス評価

### D.7.1 RealToxicityPrompts

**概要：**
- **タスク**: 有害性評価
- **サイズ**: 100K プロンプト
- **特徴**: モデルが有害なテキストを生成するかテスト

**評価指標：**
```
Toxicity Score（Perspective API使用）
- Toxicity
- Severe Toxicity
- Identity Attack
- Insult
- Profanity
- Threat
```

**入手先：**
- https://allenai.org/data/real-toxicity-prompts

### D.7.2 BOLD (Bias in Open-ended Language Generation Dataset)

**概要：**
- **タスク**: 社会的バイアス評価
- **カテゴリー**: 人種、性別、宗教、政治、職業
- **サイズ**: 23,679プロンプト

**評価：**
```
生成テキストにおける:
- ステレオタイプの頻度
- センチメントの偏り
- 表現の多様性
```

**入手先：**
- https://github.com/amazon-research/bold

---

## まとめ

### ベンチマークの選択ガイド

**言語モデリング評価:**
- 小規模: Penn Treebank
- 大規模: WikiText-103, The Pile

**理解タスク:**
- 包括的: GLUE, SuperGLUE
- 質問応答: SQuAD
- 多言語: XTREME

**生成タスク:**
- 要約: CNN/DailyMail, XSum
- 翻訳: WMT

**推論タスク:**
- 常識: HellaSwag, PIQA
- 科学: ARC
- 総合: MMLU, BIG-bench

**特殊評価:**
- コード: HumanEval
- 数学: GSM8K
- 事実性: TruthfulQA
- 安全性: RealToxicityPrompts, BOLD

### データセット利用の注意点

**ライセンス:**
- 各データセットのライセンスを確認
- 商用利用の可否

**汚染（Contamination）:**
- 訓練データに評価データが含まれていないか確認
- 特に大規模Web scrapeでは要注意

**評価の公平性:**
- Few-shot vs Fine-tuning で異なる設定
- プロンプトの影響を考慮

**最新情報:**
- ベンチマークは常に更新されている
- 新しいデータセットが随時公開

---

**📖 前の付録：[付録C 数値計算の実装](./付録C_数値計算の実装.md)**  
**📖 次の付録：[付録E 参考文献](./付録E_参考文献.md)**
