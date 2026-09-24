# Revisiting a Published NLP Study: Do Transformers Beat Classical ML on Small Datasets?

A follow-up to my own published research — fine-tuning a pretrained transformer (BanglaBERT) on the same dataset used in my co-authored paper, to test whether a modern transformer actually outperforms the classical ML and DL models we originally used.

**Original paper:** Kobra, Kobra, **Sarkar Sammi**, Rahman, Khushbu, Islam. *"Multihead Text Mining from COVID-19 Feedback Using Machine Learning, Deep Learning, and Hybrid Deep Learning Approaches."* Journal of Sensors, Wiley, 2024. [DOI: 10.1155/2024/3027199](https://doi.org/10.1155/2024/3027199)

---

## Motivation

The original study compared 8 classical ML algorithms and 5 deep learning architectures (LSTM, BiLSTM, CNN, and two hybrids) on a Bengali text classification task, achieving accuracies of 92.5–98.75%. Table 12 of the paper benchmarks these results against transformer-based studies — but those benchmarks come from **different papers on different datasets**, not a transformer applied to *this* dataset. This project closes that gap: fine-tuning BanglaBERT directly on the same 400-response dataset, under the same task, for a genuine apples-to-apples comparison.

## Dataset

- **Source:** Original survey data from the published study (self-administered questionnaires, 400 Bangladeshi students aged 10+)
- **Size:** 400 rows, perfectly balanced (200 Positive / 200 Negative)
- **Structure:** 3 text columns (impact on academic life, mental health, social life) + 1 shared sentiment label
- **Language:** Bengali (Bangla)

## Method

- **Model:** [BanglaBERT](https://huggingface.co/sagorsarker/bangla-bert-base) — a BERT model pretrained on Bengali text, fine-tuned separately on each of the three text columns to predict the same Positive/Negative label
- **Evaluation:** 5-fold stratified cross-validation (chosen deliberately over a single train/test split — with only 80 examples in a single test fold, one split can easily be a lucky or unlucky draw; this is a more rigorous evaluation than the original paper's own single-split methodology)
- **Training:** 4 epochs, learning rate 2e-5, batch size 16, per fold

## Results

| Column | BanglaBERT (mean ± std) | Best Classical ML (original paper) | Best DL (original paper) |
|---|---|---|---|
| Academic | 91.75% ± 1.70 | SGD — 95.00% | BiLSTM — 92.50% |
| Mental Health | 90.75% ± 1.27 | KNN — 93.75% | LSTM — 98.75% |
| Social | 90.75% ± 3.32 | SGD / Multi-NB — 95.00% | BiLSTM / CNN / CNN-LSTM — 92.50% |

**BanglaBERT did not outperform the classical ML models on any of the three tasks**, and performed roughly on par with (slightly below) the original deep learning architectures.

## Discussion

This is a genuinely useful negative result, not a failed experiment. It demonstrates a well-known but often overlooked principle in applied NLP: **a large pretrained transformer needs a meaningful volume of fine-tuning data to leverage its capacity advantage.** With only 320 training examples per fold, BanglaBERT (110M+ parameters) could not out-learn classical models like SGD and KNN, which are far better suited to small, clean, well-labeled datasets. Notably, this echoes a finding already present in the original paper: Table 9 shows the CNN and CNN-LSTM models scoring *negative* R² values on this same dataset — a similar case of model complexity outrunning available data.

The social-life column showed the highest variance across folds (±3.32 points, an 8.8-point swing between the best and worst fold), suggesting the model's performance on that specific task is less stable — likely reflecting greater class overlap in that column's text, a limitation worth further investigation rather than a modeling shortcoming.

**Practical takeaway:** for small, well-curated survey datasets like this one, classical ML remains a legitimate and often better first choice than reaching for a transformer by default — a transformer's advantage tends to show up at larger data scales.

## Tech Stack

- Python, pandas, scikit-learn
- Hugging Face `transformers`, `datasets`, PyTorch
- Google Colab (T4 GPU)

## Project Structure

```
├── BehaviourDataset.csv                       # Original dataset from the published study
├── banglabert_finetune.ipynb                  # Main fine-tuning + 5-fold CV script
├── banglabert_cv_vs_thesis_comparison.csv     # Final results table
└── README.md
```

## About Me

**Samrina Sarkar Sammi** — M2 Data Science & Network Intelligence student, Télécom SudParis. Co-author of the original published study this project revisits.

[LinkedIn](https://www.linkedin.com/in/samrina-sarkar-sammi-a8b716424/) · [GitHub](https://github.com/samrinasarkar-sammi) · samrinasarkar@gmail.com
