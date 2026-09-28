# Review-Intelligence-# Review Intelligence Engine

Fine-tuned RoBERTa for sentiment classification, plus a rule-based layer that buckets negative reviews into complaint categories.

## Problem Statement

Companies get too many reviews to read by hand, but real patterns are buried in there. Teams at places like American Express use classifiers to automatically flag negative feedback and route it to the right place. This project fine-tunes RoBERTa for sentiment, then adds a keyword-based categorizer for shipping/quality/customer service/pricing complaints.

## Dataset

[Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) — 100,000 labeled reviews sampled from 3.6 million available, roughly balanced between positive and negative.

## What It Builds

- A RoBERTa-base model fine-tuned for positive/negative sentiment classification
- Accuracy, F1, and a full classification report evaluation
- A keyword-based complaint categorizer (shipping / quality / customer service / pricing)
- A combined pipeline: predict sentiment, then categorize if negative

## Results (from an actual training run)

| Metric | Value |
|---|---|
| Training reviews | 100,000 (50,231 negative / 49,769 positive) |
| **Eval accuracy** | **95.64%** |
| **Eval F1** | **95.60%** |
| Eval loss | 0.226 |
| Training time | on a free Colab T4 |

95.6% accuracy on 100K real Amazon reviews is a genuinely strong, resume-worthy result for a straightforward fine-tune.

## Tech Stack

Python · HuggingFace Transformers (RoBERTa) · scikit-learn · Google Colab (free T4 GPU)

## How to Run

1. Open the notebook in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom
4. Enter a free [Groq API key](https://console.groq.com/keys) when prompted (used for an optional AI helper, not required for the core training pipeline)

## Repo Structure

```
review-intelligence-engine/
├── Review_Intelligence_Engine.ipynb
└── README.md
```

## Disclaimer

Built as a learning/portfolio project. The complaint categorizer is a simple keyword-matching heuristic, not a trained classifier — a good extension point for future work.

---
By Akshat Kesharwani
