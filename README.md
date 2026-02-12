# FinStab: A Systematic Study of Cross-Task Representation Stability in Financial Embedding Models

FinStab is a comprehensive and reproducible benchmark designed to systematically evaluate financial text embedding models across multiple task dimensions.  
It focuses on assessing embedding models in real-world financial NLP scenarios, emphasizing not only performance, but also cross-task stability and deployment efficiency.

This repository contains the evaluation framework design, benchmarking pipeline, metric implementation, and analysis modules used in our ACL submission.

---

## Overview

Financial NLP systems operate in high-stakes environments such as risk monitoring, compliance auditing, financial retrieval, and regulatory analysis. Text embeddings serve as the foundation for these systems.

---

## Tasks

FinStab evaluates embedding models on three core financial NLP tasks, each representing common industry deployment scenarios.

### 1. Financial Information Retrieval

Evaluate document retrieval capability in financial contexts such as:

- Financial statement search  
- QA recall stage in RAG systems  
- Regulatory document matching  

### 2. Financial Semantic Similarity

Measure the ability to compare deep semantic equivalence between financial texts.

**Typical scenarios:**  
- Risk disclosure comparison  
- Compliance text alignment  
- Financial statement wording evolution  

### 3. Financial Text Classification

Evaluate embedding representations as features for downstream tasks:

- Sentiment analysis (bullish / bearish)  
- Risk identification  
- Compliance detection  

---

## Stability Metrics

We introduce explicit cross-task stability quantification.

---

## Efficiency Metrics

To reflect real deployment feasibility, we consider model parameter count and maximum VRAM usage during evaluation.

---

## System Architecture

FinStab follows a **modular design**. The architecture ensures scalability and full reproducibility.

---

## Dataset

FinStab covers **30 financial datasets** across three task categories.

### Financial Information Retrieval
- FinanceBench  
- FiQA  
- Eliem Financial Reports  
- FiQA-reranking  
- Daloopa FinRetrieval  
- Additional proprietary datasets  

### Financial Semantic Similarity
- AFQMC  
- bq_corpus  
- FinMTEB  
- FinSTS  
- FinSTSb  
- FinParaSTS  
- Additional curated datasets  

### Financial Text Classification
- AdaptLLM-FiQA-SA  
- Financial text combo classification  
- Zeroshot Twitter financial news topic  
- Additional datasets  

#### Information of datasets
| Mission                         | Dataset Name                        | Source       | Number of Samples | Language |
|:--------------------------------|:------------------------------------|:-------------|------------------:|:---------|
| Financial Information Retrieval | daloopa_finretrieval                | Public       | 150              | EN       |
| Financial Information Retrieval | financebench-Subset                | Public       | 150              | EN       |
| Financial Information Retrieval | FinCorpus-Subset                  | Public       | 498043           | ZH       |
| Financial Information Retrieval | fiqa                               | Public       | 57638            | EN       |
| Financial Information Retrieval | FiQA-reranking                    | Public       | 135069           | EN       |
| Financial Information Retrieval | llamafactory-fiqa                 | Public       | 5500             | EN       |
| Financial Information Retrieval | SEC-QA-sorted-chunks             | Public       | 3755             | EN       |
| Financial Information Retrieval | TAT-QA                            | Public       | 16552            | EN       |
| Financial Information Retrieval | finInfoSearch-zh-01              | Self-written | 1045             | ZH       |
| Financial Information Retrieval | FinQA-zh-01                      | Self-written | 1016             | ZH       |
| Financial STS                   | AFQMC                             | Public       | 100000           | ZH       |
| Financial STS                   | bq_corpus                        | Public       | 100000           | ZH       |
| Financial STS                   | FinanceMTEB                      | Public       | 400              | EN       |
| Financial STS                   | FinParaSTS                       | Public       | 370              | EN       |
| Financial STS                   | FinSTS-Subset                    | Public       | 370              | EN       |
| Financial STS                   | FinSTSb                          | Public       | 2001             | EN       |
| Financial STS                   | changesInRiskDescription-en-01   | Self-written | 1101             | EN       |
| Financial STS                   | complianceTextComparison-zh-01   | Self-written | 2628             | ZH       |
| Financial Text Classification   | AdaptLLM-FiQA-SA                 | Public       | 938              | EN       |
| Financial Text Classification   | amitkediaFinancial-Fraud-Dataset | Public       | 171              | EN       |
| Financial Text Classification   | Chinese-Financial-News-Sentiment-Analysis | Public | 5000           | ZH       |
| Financial Text Classification   | CLNagisaFinancialSentimentAnalysis | Public     | 5843             | EN       |
| Financial Text Classification   | financial-news-sentiment         | Public       | 2330             | ZH       |
| Financial Text Classification   | fiqa-2018                        | Public       | 962              | EN       |
| Financial Text Classification   | twitter-financial-news-sentiment | Public       | 9939             | EN       |
| Financial Text Classification   | zeroshottwitter-financial-news-topic | Public   | 16991            | EN       |
| Financial Text Classification   | complianceClassification-en-01   | Self-written | 1147             | EN       |
| Financial Text Classification   | sentimentAnalysis-en-01          | Self-written | 1061             | EN       |

---

## Evaluated Models

We benchmark **representative embedding models** across four categories:

**Traditional Encoder Models**  
**Mainstream Open-Source Models**  
**Financial Domain-Specific Models**  
**Commercial / Semi-Commercial Models**  

All models are **locally deployed unless API-based**.

---

## Repository Structure
```
.
├── config.json   #Configuration file for the evaluation script
├── evaluate.py   #Dataset evaluation script
├── README.md
├── socreCalculator.py   #Final Score Calculation
├── datasets   #Self-built dataset
│   ├── class_1
│   │   ├── class1-finInfoSearch-zh-01
│   │   │   └── train.csv
│   │   └── class1-FinQA-zh-01
│   │       └── train.csv
│   ├── class_2
│   │   ├── class2-changesInRiskDescription-en-01
│   │   │   └── train.csv
│   │   └── class2-complianceTextComparison-zh-01
│   │       └── train.csv
│   └── class_3
│       ├── class3-complianceClassification-en-01
│       │   └── train.csv
│       └── class3-sentimentAnalysis-en-01
│           └── train.csv
└── exampleScore  
    ├── pareto_class1.png
    ├── pareto_class2.png
    ├── pareto_class3.png
    └── scores.json
```
