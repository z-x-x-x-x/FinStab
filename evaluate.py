import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import torch
from sentence_transformers import SentenceTransformer, util
import warnings
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from enum import Enum
import logging
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, average_precision_score, classification_report
import joblib
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置加载 ====================
def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info("配置文件加载成功")
        return config
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"配置文件解析失败: {e}")
        raise

# ==================== 公共枚举和类定义 ====================
class DatasetType(Enum):
    # Class 1 类型
    FINANCE_BENCH = "finance_bench"
    FIQA = "fiqa"
    ADAPT_LLM = "adapt_llm"
    OPEN_FIN_QA = "open_fin_qa"
    FIQA_RERANKING = "fiqa_reranking"
    ADAPT_LLM_FIQA_SA = "adapt_llm_fiqa_sa"
    ELIEM_FINANCIAL_REPORTS = "eliem_financial_reports"
    DALOOPA_FINRETRIEVAL = "daloopa_finretrieval"
    LLAMAFACTORY_FIQA = "llamafactory_fiqa"
    SEC_QA_SORTED_CHUNKS = "sec_qa_sorted_chunks"
    FINCORPUS = "fincorpus"
    TAT_QA = "tat_qa"
    FIN_INFO_SEARCH_ZH = "fin_info_search_zh"
    FIN_QA_ZH = "fin_qa_zh"
    
    # Class 3 类型
    FINANCIAL_PHRASEBANK = "financial_phrasebank"
    FINCHINA_SENTIMENT = "finchina_sentiment"
    FINANCIAL_FRAUD = "financial_fraud"
    NICKMUCHI_FINANCIAL_TEXT = "nickmuchifinancial_text"
    TWITTER_FINANCIAL_NEWS_TOPIC = "twitter_financial_news_topic"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_IDENTIFICATION = "risk_identification"
    COMPLIANCE_CLASSIFICATION = "compliance_classification"

@dataclass
class DatasetConfig:
    name: str
    type: DatasetType
    paths: Dict[str, str]
    description: str = ""
    options: Dict[str, Any] = field(default_factory=dict)
    sample_size: Optional[int] = None
    task_type: str = "classification"

# ==================== Class 1: 金融检索评估 ====================
@dataclass
class RetrievalResult:
    query_id: str
    query: str
    retrieved_docs: List[str]
    retrieved_doc_ids: List[str]
    scores: List[float]
    relevant_docs: List[str]
    relevance_scores: List[int]

class FinancialRetrievalEvaluator:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"加载模型: {model_path}")
        self.model = SentenceTransformer(model_path, device=device)
        self.device = device
        logger.info(f"模型加载完成，设备: {device}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def search(self, 
               query: str, 
               corpus_embeddings: np.ndarray,
               corpus_ids: List[str],
               top_k: int = 10) -> Tuple[List[str], List[float]]:
        query_embedding = self.model.encode(
            query, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(similarities, k=min(top_k, len(corpus_ids)))
        
        retrieved_ids = [corpus_ids[idx] for idx in top_results.indices.tolist()]
        scores = top_results.values.tolist()
        
        return retrieved_ids, scores
    
    def evaluate_retrieval(self,
                          queries: Dict[str, str],
                          corpus: Dict[str, str],
                          qrels: Dict[str, Dict[str, int]],
                          top_k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        logger.info("开始评估检索性能...")
        
        corpus_ids = list(corpus.keys())
        corpus_texts = list(corpus.values())
        
        logger.info(f"编码文档库 ({len(corpus_texts)} 个文档)...")
        corpus_embeddings = self.encode_texts(corpus_texts)
        
        all_metrics = {}
        
        for k in top_k_values:
            recalls = []
            ndcgs = []
            
            logger.info(f"评估 Recall@{k} 和 nDCG@{k}...")
            processed_count = 0
            
            for query_id, query_text in tqdm(queries.items(), desc=f"处理查询 (k={k})"):
                if query_id not in qrels or not qrels[query_id]:
                    continue
                    
                retrieved_ids, scores = self.search(
                    query_text, corpus_embeddings, corpus_ids, top_k=k
                )
                
                relevant_docs = qrels.get(query_id, {})
                if not relevant_docs:
                    continue
                
                processed_count += 1
                
                relevant_retrieved = sum(1 for doc_id in retrieved_ids 
                                       if doc_id in relevant_docs and relevant_docs[doc_id] > 0)
                recall = relevant_retrieved / len(relevant_docs)
                recalls.append(recall)
                
                dcg = 0.0
                for i, doc_id in enumerate(retrieved_ids[:k]):
                    rel = relevant_docs.get(doc_id, 0)
                    if rel > 0:
                        dcg += (2**rel - 1) / np.log2(i + 2)
                
                ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]
                idcg = sum((2**rel - 1) / np.log2(i + 2) 
                          for i, rel in enumerate(ideal_relevances))
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
            
            logger.info(f"实际处理的查询数: {processed_count}")
            if recalls and processed_count > 0:
                all_metrics[f"Recall@{k}"] = np.mean(recalls)
                all_metrics[f"nDCG@{k}"] = np.mean(ndcgs)
                all_metrics[f"num_queries_processed@{k}"] = processed_count
                all_metrics[f"total_queries"] = len(queries)
        
        return all_metrics

class Class1DatasetLoader:
    @staticmethod
    def load_fincorpus(file_path: str, sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        """加载FinCorpus数据集"""
        queries = {}
        corpus = {}
        qrels = {}
        
        logger.info(f"加载FinCorpus数据集: {file_path}")
        
        try:
            if file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        try:
                            item = json.loads(line.strip())
                            text = item.get("text", "")
                            if "答案：" in text and "分析解释：" in text:
                                # 分割问题和答案
                                parts = text.split("答案：")
                                if len(parts) >= 2:
                                    question_part = parts[0].strip()
                                    answer_part = parts[1].split("分析解释：")[0].strip()
                                    
                                    query_id = f"query_{i}"
                                    doc_id = f"doc_{i}"
                                    
                                    queries[query_id] = question_part
                                    corpus[doc_id] = answer_part
                                    qrels[query_id] = {doc_id: 1}
                        except json.JSONDecodeError:
                            logger.warning(f"跳过第{i}行，JSON解析失败")
            
            if sample_size and len(queries) > sample_size:
                queries, corpus, qrels = Class1DatasetLoader._sample_dataset(
                    queries, corpus, qrels, sample_size
                )
            
            logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档")
            return queries, corpus, qrels
            
        except Exception as e:
            logger.error(f"加载FinCorpus数据集失败: {e}")
            return {}, {}, {}
    
    @staticmethod
    def load_tat_qa(file_path: str, sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        """加载TAT-QA数据集"""
        queries = {}
        corpus = {}
        qrels = {}
        
        logger.info(f"加载TAT-QA数据集: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item_idx, item in enumerate(data):
                # 提取表格和段落文本作为文档
                table_text = ""
                if "table" in item and "table" in item["table"]:
                    for row in item["table"]["table"]:
                        table_text += " ".join([str(cell) for cell in row]) + " "
                
                paragraphs_text = ""
                if "paragraphs" in item:
                    for para in item["paragraphs"]:
                        paragraphs_text += para["text"] + " "
                
                # 合并表格和段落作为文档
                doc_id = f"doc_{item_idx}"
                corpus[doc_id] = table_text + " " + paragraphs_text
                
                # 提取问题作为查询
                if "questions" in item:
                    for q_idx, question_item in enumerate(item["questions"]):
                        query_id = f"query_{item_idx}_{q_idx}"
                        queries[query_id] = question_item["question"]
                        qrels[query_id] = {doc_id: 1}
            
            if sample_size and len(queries) > sample_size:
                queries, corpus, qrels = Class1DatasetLoader._sample_dataset(
                    queries, corpus, qrels, sample_size
                )
            
            logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档")
            return queries, corpus, qrels
            
        except Exception as e:
            logger.error(f"加载TAT-QA数据集失败: {e}")
            return {}, {}, {}
    
    @staticmethod
    def load_fin_info_search_zh(file_path: str, 
                               query_field: str = "query",
                               doc_field: str = "doc_text", 
                               label_field: str = "label",
                               sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        """加载中文金融信息检索数据集"""
        queries = {}
        corpus = {}
        qrels = {}
        
        logger.info(f"加载FinInfoSearch中文数据集: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            for i, row in df.iterrows():
                if query_field in df.columns and doc_field in df.columns and label_field in df.columns:
                    query_id = f"query_{i}"
                    doc_id = f"doc_{i}"
                    
                    query_text = str(row[query_field])
                    doc_text = str(row[doc_field])
                    label = int(row[label_field])
                    
                    if query_text and doc_text and len(query_text) > 5 and len(doc_text) > 5:
                        queries[query_id] = query_text
                        corpus[doc_id] = doc_text
                        
                        if label == 1:  # 只添加正例
                            if query_id not in qrels:
                                qrels[query_id] = {}
                            qrels[query_id][doc_id] = 1
            
            if sample_size and len(queries) > sample_size:
                queries, corpus, qrels = Class1DatasetLoader._sample_dataset(
                    queries, corpus, qrels, sample_size
                )
            
            logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档, 正例对: {sum(len(v) for v in qrels.values())}")
            return queries, corpus, qrels
            
        except Exception as e:
            logger.error(f"加载FinInfoSearch中文数据集失败: {e}")
            return {}, {}, {}
    
    @staticmethod
    def load_fin_qa_zh(file_path: str,
                      question_field: str = "question",
                      context_field: str = "context",
                      answer_field: str = "answer",
                      sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        """加载中文金融问答数据集"""
        queries = {}
        corpus = {}
        qrels = {}
        
        logger.info(f"加载FinQA中文数据集: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            for i, row in df.iterrows():
                if question_field in df.columns and context_field in df.columns:
                    query_id = f"query_{i}"
                    doc_id = f"doc_{i}"
                    
                    question_text = str(row[question_field])
                    context_text = str(row[context_field])
                    
                    if question_text and context_text and len(question_text) > 5 and len(context_text) > 5:
                        queries[query_id] = question_text
                        corpus[doc_id] = context_text
                        qrels[query_id] = {doc_id: 1}
            
            if sample_size and len(queries) > sample_size:
                queries, corpus, qrels = Class1DatasetLoader._sample_dataset(
                    queries, corpus, qrels, sample_size
                )
            
            logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档")
            return queries, corpus, qrels
            
        except Exception as e:
            logger.error(f"加载FinQA中文数据集失败: {e}")
            return {}, {}, {}
    
    @staticmethod
    def load_question_answer_pairs(file_path: str, 
                                  question_field: str,
                                  answer_field: str,
                                  sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        queries = {}
        corpus = {}
        qrels = {}
        
        logger.info(f"加载问答对数据集: {file_path}")
        logger.info(f"问题字段: {question_field}, 答案字段: {answer_field}")
        
        try:
            if file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        try:
                            item = json.loads(line.strip())
                            query_id = f"query_{i}"
                            doc_id = f"doc_{i}"
                            
                            question = item.get(question_field, "")
                            answer = item.get(answer_field, "")
                            
                            if question and answer and len(question) > 5 and len(answer) > 5:
                                queries[query_id] = str(question)
                                corpus[doc_id] = str(answer)
                                qrels[query_id] = {doc_id: 1}
                        except json.JSONDecodeError:
                            logger.warning(f"跳过第{i}行，JSON解析失败")
            
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
                
                for i, item in enumerate(items):
                    query_id = f"query_{i}"
                    doc_id = f"doc_{i}"
                    
                    question = item.get(question_field, "")
                    answer = item.get(answer_field, "")
                    
                    if question and answer and len(question) > 5 and len(answer) > 5:
                        queries[query_id] = str(question)
                        corpus[doc_id] = str(answer)
                        qrels[query_id] = {doc_id: 1}
            
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                
                for i, row in df.iterrows():
                    query_id = f"query_{i}"
                    doc_id = f"doc_{i}"
                    
                    question = str(row[question_field]) if question_field in df.columns else ""
                    answer = str(row[answer_field]) if answer_field in df.columns else ""
                    
                    if question and answer and len(question) > 5 and len(answer) > 5:
                        queries[query_id] = question
                        corpus[doc_id] = answer
                        qrels[query_id] = {doc_id: 1}
        
        except Exception as e:
            logger.error(f"加载问答对数据集失败: {e}")
            return {}, {}, {}
        
        if sample_size and len(queries) > sample_size:
            queries, corpus, qrels = Class1DatasetLoader._sample_dataset(
                queries, corpus, qrels, sample_size
            )
        
        logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档")
        return queries, corpus, qrels
    
    @staticmethod
    def load_input_output_pairs(file_path: str,
                               input_field: str,
                               output_field: str,
                               sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        return Class1DatasetLoader.load_question_answer_pairs(
            file_path, input_field, output_field, sample_size
        )
    
    @staticmethod
    def load_standard_retrieval(corpus_path: str, 
                               queries_path: str = None, 
                               qrels_path: str = None,
                               sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        corpus = {}
        queries = {}
        qrels = {}
        
        if corpus_path and os.path.exists(corpus_path):
            logger.info(f"加载语料库: {corpus_path}")
            if corpus_path.endswith('.jsonl'):
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        doc_id = item.get("_id", str(len(corpus)))
                        text = item.get("text", "")
                        if text:
                            corpus[doc_id] = text
        
        if queries_path and os.path.exists(queries_path):
            logger.info(f"加载查询: {queries_path}")
            if queries_path.endswith('.jsonl'):
                with open(queries_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        query_id = item.get("_id", str(len(queries)))
                        queries[query_id] = item.get("text", "")
        
        if qrels_path and os.path.exists(qrels_path):
            logger.info(f"加载qrels: {qrels_path}")
            Class1DatasetLoader._load_qrels(qrels_path, qrels, corpus)
        
        if not qrels and queries:
            logger.warning("未找到qrels文件，创建简单qrels用于测试")
            Class1DatasetLoader._create_test_qrels(queries, corpus, qrels)
        
        if sample_size and len(queries) > sample_size:
            queries, corpus, qrels = Class1DatasetLoader._sample_dataset(
                queries, corpus, qrels, sample_size
            )
        
        logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档")
        return queries, corpus, qrels
    
    @staticmethod
    def load_reranking_pairs(file_path: str,
                            query_field: str = "query",
                            positive_field: str = "positive",
                            negative_field: str = "negative",
                            sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        queries = {}
        corpus = {}
        qrels = {}
        
        logger.info(f"加载重排序数据集: {file_path}")
        
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                logger.error(f"不支持的格式: {file_path}")
                return {}, {}, {}
            
            for i, row in df.iterrows():
                query_id = f"query_{i}"
                query_text = str(row[query_field])
                positive_text = str(row[positive_field])
                negative_text = str(row[negative_field]) if negative_field in df.columns else ""
                
                queries[query_id] = query_text
                
                positive_id = f"pos_{i}"
                corpus[positive_id] = positive_text
                qrels[query_id] = {positive_id: 1}
                
                if negative_text:
                    negative_id = f"neg_{i}"
                    corpus[negative_id] = negative_text
                    qrels[query_id][negative_id] = 0
        
        except Exception as e:
            logger.error(f"加载重排序数据集失败: {e}")
            return {}, {}, {}
        
        if sample_size and len(queries) > sample_size:
            queries, corpus, qrels = Class1DatasetLoader._sample_dataset(
                queries, corpus, qrels, sample_size
            )
        
        logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档")
        return queries, corpus, qrels
    
    @staticmethod
    def load_financebench(file_path: str, sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        queries = {}
        corpus = {}
        qrels = {}
        
        logger.info(f"加载FinanceBench数据集: {file_path}")
        
        try:
            if file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        query_id = item.get("financebench_id", str(len(queries)))
                        queries[query_id] = item["question"]
                        
                        for evidence in item.get("evidence", []):
                            doc_id = f"{query_id}_evidence_{len(corpus)}"
                            corpus[doc_id] = evidence["evidence_text"]
                            
                            if query_id not in qrels:
                                qrels[query_id] = {}
                            qrels[query_id][doc_id] = 1
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        items = data
                    else:
                        items = [data]
                    
                    for i, item in enumerate(items):
                        query_id = item.get("financebench_id", f"query_{i}")
                        queries[query_id] = item["question"]
                        
                        for evidence in item.get("evidence", []):
                            doc_id = f"{query_id}_evidence_{len(corpus)}"
                            corpus[doc_id] = evidence["evidence_text"]
                            
                            if query_id not in qrels:
                                qrels[query_id] = {}
                            qrels[query_id][doc_id] = 1
        
        except Exception as e:
            logger.error(f"加载FinanceBench数据集失败: {e}")
            raise
        
        if sample_size and len(queries) > sample_size:
            queries, corpus, qrels = Class1DatasetLoader._sample_dataset(
                queries, corpus, qrels, sample_size
            )
        
        logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档")
        return queries, corpus, qrels
    
    @staticmethod
    def load_daloopa_finretrieval(base_path: str, sample_size: Optional[int] = None) -> Tuple[Dict, Dict, Dict]:
        queries = {}
        corpus = {}
        qrels = {}
        
        logger.info(f"加载daloopa_finretrieval数据集，基础路径: {base_path}")
        
        questions_path = os.path.join(base_path, "questions.parquet")
        responses_path = os.path.join(base_path, "responses.parquet")
        
        if not os.path.exists(questions_path):
            logger.error(f"questions.parquet文件不存在: {questions_path}")
            return {}, {}, {}
        
        try:
            questions_df = pd.read_parquet(questions_path)
            logger.info(f"questions.parquet加载完成，形状: {questions_df.shape}")
            
            for i, row in questions_df.iterrows():
                query_id = f"query_{i}"
                question = str(row['question']) if 'question' in row else ""
                answer = str(row['answer']) if 'answer' in row and pd.notna(row['answer']) else ""
                
                queries[query_id] = question
                
                if answer and len(answer) > 5:
                    doc_id = f"q_ans_{i}"
                    corpus[doc_id] = answer
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = 1
            
            if os.path.exists(responses_path):
                responses_df = pd.read_parquet(responses_path)
                logger.info(f"responses.parquet加载完成，形状: {responses_df.shape}")
                
                for i, row in responses_df.iterrows():
                    doc_id = f"r_{i}"
                    response = str(row['response']) if 'response' in row else ""
                    
                    if response and len(response) > 5:
                        corpus[doc_id] = response
        
        except Exception as e:
            logger.error(f"加载daloopa_finretrieval数据集失败: {e}")
            return {}, {}, {}
        
        if not qrels and len(queries) > 0:
            Class1DatasetLoader._create_keyword_based_qrels(queries, corpus, qrels)
        
        if sample_size and len(queries) > sample_size:
            queries, corpus, qrels = Class1DatasetLoader._sample_dataset(
                queries, corpus, qrels, sample_size
            )
        
        logger.info(f"加载完成: {len(queries)} 个查询, {len(corpus)} 个文档")
        return queries, corpus, qrels
    
    @staticmethod
    def _load_qrels(qrels_path: str, qrels: Dict, corpus: Dict):
        if qrels_path.endswith('.jsonl'):
            with open(qrels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    query_id = item.get("query_id")
                    if not query_id:
                        continue
                    
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    
                    positive_passages = item.get("positive_passages", [])
                    for passage in positive_passages:
                        doc_id = passage.get("doc_id")
                        if doc_id and doc_id in corpus:
                            qrels[query_id][doc_id] = 1
                    
                    negative_passages = item.get("negative_passages", [])
                    for passage in negative_passages:
                        doc_id = passage.get("doc_id")
                        if doc_id and doc_id in corpus:
                            qrels[query_id][doc_id] = 0
        elif qrels_path.endswith('.tsv'):
            with open(qrels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        query_id, doc_id, relevance = parts[0], parts[1], int(parts[2])
                        if query_id not in qrels:
                            qrels[query_id] = {}
                        qrels[query_id][doc_id] = relevance
    
    @staticmethod
    def _create_test_qrels(queries: Dict, corpus: Dict, qrels: Dict, max_per_query: int = 3):
        corpus_keys = list(corpus.keys())
        if not corpus_keys:
            return
        
        for query_id in list(queries.keys())[:100]:
            relevant_docs = random.sample(corpus_keys, min(max_per_query, len(corpus_keys)))
            qrels[query_id] = {doc_id: 1 for doc_id in relevant_docs}
    
    @staticmethod
    def _create_keyword_based_qrels(queries: Dict, corpus: Dict, qrels: Dict):
        for query_id, query_text in queries.items():
            if not query_text:
                continue
            
            keywords = [word for word in query_text.split() if len(word) > 3][:3]
            if not keywords:
                continue
            
            for doc_id, doc_text in corpus.items():
                if any(keyword.lower() in doc_text.lower() for keyword in keywords):
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = 1
                    break
    
    @staticmethod
    def _sample_dataset(queries: Dict, corpus: Dict, qrels: Dict, sample_size: int) -> Tuple[Dict, Dict, Dict]:
        logger.info(f"数据集过大，采样 {sample_size} 个查询")
        all_query_ids = list(queries.keys())
        sampled_query_ids = random.sample(all_query_ids, min(sample_size, len(all_query_ids)))
        
        sampled_queries = {qid: queries[qid] for qid in sampled_query_ids}
        sampled_qrels = {qid: qrels[qid] for qid in sampled_query_ids if qid in qrels}
        
        sampled_corpus = {}
        for qid in sampled_query_ids:
            if qid in qrels:
                for doc_id in qrels[qid]:
                    if doc_id in corpus:
                        sampled_corpus[doc_id] = corpus[doc_id]
        
        return sampled_queries, sampled_corpus, sampled_qrels
    
    @staticmethod
    def load_dataset(dataset_config: Dict) -> Tuple[Dict, Dict, Dict]:
        dataset_type = DatasetType(dataset_config.get("type", "FINANCE_BENCH").lower())
        sample_size = dataset_config.get("sample_size", 3000)
        options = dataset_config.get("options", {})
        
        if dataset_type == DatasetType.FINANCE_BENCH:
            return Class1DatasetLoader.load_financebench(
                dataset_config["paths"].get("main"), sample_size
            )
        
        elif dataset_type == DatasetType.FIQA:
            return Class1DatasetLoader.load_standard_retrieval(
                dataset_config["paths"].get("corpus"),
                dataset_config["paths"].get("queries"),
                dataset_config["paths"].get("qrels"),
                sample_size
            )
        
        elif dataset_type == DatasetType.FIQA_RERANKING:
            return Class1DatasetLoader.load_reranking_pairs(
                dataset_config["paths"].get("main"),
                options.get("query_field", "query"),
                options.get("positive_field", "positive"),
                options.get("negative_field", "negative"),
                sample_size
            )
        
        elif dataset_type == DatasetType.DALOOPA_FINRETRIEVAL:
            return Class1DatasetLoader.load_daloopa_finretrieval(
                dataset_config["paths"].get("main"), sample_size
            )
        
        elif dataset_type == DatasetType.LLAMAFACTORY_FIQA:
            return Class1DatasetLoader.load_input_output_pairs(
                dataset_config["paths"].get("main"),
                options.get("input_field", "input"),
                options.get("output_field", "output"),
                sample_size
            )
        
        elif dataset_type == DatasetType.SEC_QA_SORTED_CHUNKS:
            return Class1DatasetLoader.load_question_answer_pairs(
                dataset_config["paths"].get("main"),
                options.get("question_field", "questions"),
                options.get("answer_field", "answers"),
                sample_size
            )
        
        elif dataset_type == DatasetType.FINCORPUS:
            return Class1DatasetLoader.load_fincorpus(
                dataset_config["paths"].get("main"), sample_size
            )
        
        elif dataset_type == DatasetType.TAT_QA:
            return Class1DatasetLoader.load_tat_qa(
                dataset_config["paths"].get("main"), sample_size
            )
        
        elif dataset_type == DatasetType.FIN_INFO_SEARCH_ZH:
            return Class1DatasetLoader.load_fin_info_search_zh(
                dataset_config["paths"].get("main"),
                options.get("query_field", "query"),
                options.get("doc_field", "doc_text"),
                options.get("label_field", "label"),
                sample_size
            )
        
        elif dataset_type == DatasetType.FIN_QA_ZH:
            return Class1DatasetLoader.load_fin_qa_zh(
                dataset_config["paths"].get("main"),
                options.get("question_field", "question"),
                options.get("context_field", "context"),
                options.get("answer_field", "answer"),
                sample_size
            )
        
        else:
            logger.warning(f"不支持的数据集类型: {dataset_type}，使用默认加载方式")
            return Class1DatasetLoader.load_financebench(
                dataset_config["paths"].get("main"), sample_size
            )

# ==================== Class 2: 语义文本相似度评估 ====================
@dataclass
class STSResult:
    dataset_name: str
    pearson: float
    spearman: float
    num_samples: int
    split_name: str = "dataset"

class STSModel:
    def __init__(self, model_path):
        logger.info("开始加载模型...")
        self.model = SentenceTransformer(
            model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"模型加载完成，设备: {self.model.device}")

    def encode(self, sentences, batch_size=32, **kwargs):
        all_embeddings = []
        
        with tqdm(total=len(sentences), desc="编码句子", unit="句") as pbar:
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(embeddings)
            pbar.update(len(sentences))
        
        return np.vstack(all_embeddings)

class Class2DatasetLoader:
    @staticmethod
    def load_dataset(file_path):
        logger.info(f"加载数据集文件: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"错误: 文件不存在: {file_path}")
            return None
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.parquet':
                logger.info("检测到parquet格式，使用pandas读取...")
                df = pd.read_parquet(file_path)
                
            elif file_ext == '.arrow':
                logger.info("检测到arrow格式，使用pandas读取...")
                df = pd.read_feather(file_path)
                
            elif file_ext == '.csv':
                logger.info("检测到CSV格式，使用pandas读取...")
                df = pd.read_csv(file_path)
                
            elif file_ext == '.tsv':
                logger.info("检测到TSV格式，使用pandas读取...")
                df = pd.read_csv(file_path, sep='\t')
                
            elif file_ext == '.json':
                logger.info("检测到JSON格式，使用pandas读取...")
                df = pd.read_json(file_path)
                
            elif file_ext == '.jsonl':
                logger.info("检测到JSONL格式，使用pandas读取...")
                df = pd.read_json(file_path, lines=True)
                
            else:
                logger.error(f"错误: 不支持的文件格式: {file_ext}")
                return None
                
            logger.info(f"文件加载成功，读取到 {len(df)} 行数据")
            
            # 检查列名，适配不同数据集的列名
            sentence1_col = None
            sentence2_col = None
            score_col = None
            
            # 尝试匹配常见列名
            possible_sentence1_cols = ['sentence1', 'sent1', 'question1', 'query', 'text1', 'sent_a']
            possible_sentence2_cols = ['sentence2', 'sent2', 'question2', 'answer', 'text2', 'sent_b']
            possible_score_cols = ['score', 'label', 'similarity', 'relevance', 'gold_score']
            
            for col in df.columns:
                if col in possible_sentence1_cols:
                    sentence1_col = col
                elif col in possible_sentence2_cols:
                    sentence2_col = col
                elif col in possible_score_cols:
                    score_col = col
            
            # 如果未找到标准列名，使用第一、第二列作为句子，最后一列作为分数
            if sentence1_col is None and len(df.columns) >= 2:
                sentence1_col = df.columns[0]
            
            if sentence2_col is None and len(df.columns) >= 3:
                sentence2_col = df.columns[1]
            
            if score_col is None and len(df.columns) >= 3:
                score_col = df.columns[-1]
            
            # 对于ATEC数据集（无列名）
            if 'ATEC' in file_path and len(df.columns) >= 4:
                # ATEC格式：id, sentence1, sentence2, score
                sentence1_col = df.columns[1]
                sentence2_col = df.columns[2]
                score_col = df.columns[3]
            
            if sentence1_col and sentence2_col and score_col:
                dataset = {
                    'data': {
                        'sentence1': df[sentence1_col].astype(str).tolist(),
                        'sentence2': df[sentence2_col].astype(str).tolist(),
                        'score': df[score_col].astype(float).tolist()
                    }
                }
                logger.info(f"成功提取 {len(df)} 个样本")
                logger.info(f"使用的列: sentence1={sentence1_col}, sentence2={sentence2_col}, score={score_col}")
                
                return dataset
            else:
                logger.error(f"错误: 无法找到合适的列。可用列: {list(df.columns)}")
                return None
                
        except Exception as e:
            logger.error(f"加载文件失败: {e}")
            return None

# ==================== Class 3: 文本分类评估 ====================
class FinancialTextClassifier:
    def __init__(self, model_path: str, device: str = None, classifier_type: str = "logistic"):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"加载Sentence Transformer模型: {model_path}")
        self.model = SentenceTransformer(model_path, device=device)
        self.device = device
        self.classifier_type = classifier_type
        self.classifier = None
        self.model_name = os.path.basename(model_path)
        logger.info(f"模型加载完成，设备: {device}, 分类器类型: {classifier_type}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray):
        logger.info(f"训练{self.classifier_type}分类器...")
        
        if self.classifier_type == "logistic":
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif self.classifier_type == "svm":
            self.classifier = SVC(
                kernel='linear',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"不支持的分类器类型: {self.classifier_type}")
        
        self.classifier.fit(X_train, y_train)
        logger.info(f"分类器训练完成，训练样本数: {len(X_train)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise ValueError("分类器尚未训练")
        return self.classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise ValueError("分类器尚未训练")
        return self.classifier.predict_proba(X)
    
    def evaluate_classification(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if self.classifier is None:
            raise ValueError("分类器尚未训练")
        
        y_test_np = np.array(y_test)
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        metrics = {}
        
        f1_micro = f1_score(y_test_np, y_pred, average='micro')
        f1_macro = f1_score(y_test_np, y_pred, average='macro')
        f1_weighted = f1_score(y_test_np, y_pred, average='weighted')
        
        metrics['F1_micro'] = f1_micro
        metrics['F1_macro'] = f1_macro
        metrics['F1_weighted'] = f1_weighted
        
        unique_labels = np.unique(y_test_np)
        if len(unique_labels) == 2:
            positive_proba = y_pred_proba[:, 1]
            average_precision = average_precision_score(y_test_np, positive_proba)
            metrics['Average_Precision'] = average_precision
        else:
            ap_scores = []
            n_classes = len(unique_labels)
            for i in range(n_classes):
                y_test_binary = (y_test_np == i).astype(int)
                y_pred_proba_binary = y_pred_proba[:, i]
                if len(np.unique(y_test_binary)) > 1:
                    try:
                        ap = average_precision_score(y_test_binary, y_pred_proba_binary)
                        ap_scores.append(ap)
                    except:
                        continue
            
            if ap_scores:
                metrics['Mean_Average_Precision'] = np.mean(ap_scores)
        
        accuracy = np.mean(y_pred == y_test_np)
        metrics['Accuracy'] = accuracy
        
        cls_report = classification_report(y_test_np, y_pred, output_dict=True)
        
        return metrics, cls_report

class Class3DatasetLoader:
    @staticmethod
    def load_tsv_no_header(file_path: str, text_idx: int = 1, label_idx: int = 3, 
                          sample_size: Optional[int] = None, label_mapping: Optional[Dict[int, int]] = None) -> Tuple[List[str], List[int]]:
        texts = []
        labels = []
        
        logger.info(f"加载TSV格式数据集: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > max(text_idx, label_idx):
                        text = parts[text_idx]
                        try:
                            label = int(parts[label_idx])
                            
                            if label_mapping:
                                label = label_mapping.get(label, label)
                            
                            texts.append(text)
                            labels.append(label)
                        except ValueError:
                            continue
        except Exception as e:
            logger.error(f"加载TSV数据集失败: {e}")
            return [], []
        
        if sample_size and len(texts) > sample_size:
            logger.info(f"数据集过大，采样 {sample_size} 个样本")
            indices = random.sample(range(len(texts)), sample_size)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        logger.info(f"加载完成: {len(texts)} 个样本")
        return texts, labels
    
    @staticmethod
    def load_csv_with_columns(file_path: str, text_column: str, label_column: str, 
                             sample_size: Optional[int] = None, label_mapping: Optional[Dict[Any, int]] = None) -> Tuple[List[str], List[int]]:
        texts = []
        labels = []
        
        logger.info(f"加载CSV/Parquet数据集: {file_path}")
        
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                logger.error(f"不支持的格式: {file_path}")
                return [], []
            
            logger.info(f"数据集列名: {list(df.columns)}")
            
            if text_column not in df.columns or label_column not in df.columns:
                logger.error(f"未找到所需列: 文本列='{text_column}', 标签列='{label_column}'")
                return [], []
            
            for _, row in df.iterrows():
                text = str(row[text_column])
                label_raw = row[label_column]
                
                if pd.isna(label_raw):
                    continue
                    
                try:
                    if isinstance(label_raw, str):
                        label_raw_lower = label_raw.lower()
                        if label_raw_lower in ['positive', 'pos', '1', 'yes', 'true']:
                            label = 1
                        elif label_raw_lower in ['negative', 'neg', '0', 'no', 'false']:
                            label = 0
                        elif label_raw_lower in ['neutral', 'neut', '2']:
                            label = 2
                        else:
                            try:
                                label = int(float(label_raw))
                            except:
                                if label_mapping:
                                    label = label_mapping.get(label_raw, -1)
                                else:
                                    continue
                    else:
                        label = int(float(label_raw))
                    
                    if label != -1 and text and len(text.strip()) > 0:
                        texts.append(text)
                        labels.append(label)
                except (ValueError, TypeError) as e:
                    continue
                
        except Exception as e:
            logger.error(f"加载CSV/Parquet数据集失败: {e}")
            return [], []
        
        if sample_size and len(texts) > sample_size:
            logger.info(f"数据集过大，采样 {sample_size} 个样本")
            indices = random.sample(range(len(texts)), sample_size)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        logger.info(f"加载完成: {len(texts)} 个样本")
        return texts, labels
    
    @staticmethod
    def load_dataset(dataset_config: Dict, test_size: float = 0.2) -> Tuple[List[str], List[int], List[str], List[int]]:
        dataset_type = DatasetType(dataset_config.get("type", "SENTIMENT_ANALYSIS").lower())
        sample_size = dataset_config.get("sample_size", 3000)
        options = dataset_config.get("options", {})
        
        if dataset_type == DatasetType.ADAPT_LLM_FIQA_SA:
            label_mapping = {2: 1, 1: 0, 0: 2}
            texts, labels = Class3DatasetLoader.load_tsv_no_header(
                dataset_config["paths"].get("train"), text_idx=1, label_idx=3, 
                sample_size=sample_size, label_mapping=label_mapping
            )
        elif dataset_type == DatasetType.FINANCIAL_FRAUD:
            texts, labels = Class3DatasetLoader.load_csv_with_columns(
                dataset_config["paths"].get("train"), text_column="Fillings", label_column="Fraud", 
                sample_size=sample_size
            )
        elif dataset_type == DatasetType.SENTIMENT_ANALYSIS:
            # 处理中文金融新闻情感分析数据集
            if "chinese" in dataset_config["name"].lower():
                text_col = options.get("text_column", "text")
                label_col = options.get("label_column", "negative")
                texts, labels = Class3DatasetLoader.load_csv_with_columns(
                    dataset_config["paths"].get("train"), 
                    text_column=text_col, 
                    label_column=label_col, 
                    sample_size=sample_size
                )
            elif "clnagisa" in dataset_config["name"].lower():
                label_mapping = {"positive": 1, "negative": 0, "neutral": 2}
                texts, labels = Class3DatasetLoader.load_csv_with_columns(
                    dataset_config["paths"].get("train"), text_column="Sentence", label_column="Sentiment", 
                    sample_size=sample_size, label_mapping=label_mapping
                )
            else:
                text_col = options.get("text_column", "text")
                label_col = options.get("label_column", "label")
                texts, labels = Class3DatasetLoader.load_csv_with_columns(
                    dataset_config["paths"].get("train"), 
                    text_column=text_col, 
                    label_column=label_col, 
                    sample_size=sample_size
                )
        elif dataset_type == DatasetType.RISK_IDENTIFICATION:
            texts, labels = Class3DatasetLoader.load_csv_with_columns(
                dataset_config["paths"].get("train"), text_column="text", label_column="label", 
                sample_size=sample_size
            )
        elif dataset_type == DatasetType.COMPLIANCE_CLASSIFICATION:
            texts, labels = Class3DatasetLoader.load_csv_with_columns(
                dataset_config["paths"].get("train"), text_column="text", label_column="label", 
                sample_size=sample_size
            )
        elif dataset_type == DatasetType.NICKMUCHI_FINANCIAL_TEXT:
            texts, labels = Class3DatasetLoader.load_csv_with_columns(
                dataset_config["paths"].get("train"), text_column="text", label_column="label", 
                sample_size=sample_size
            )
        elif dataset_type == DatasetType.TWITTER_FINANCIAL_NEWS_TOPIC:
            texts, labels = Class3DatasetLoader.load_csv_with_columns(
                dataset_config["paths"].get("train"), text_column="text", label_column="label", 
                sample_size=sample_size
            )
        else:
            logger.warning(f"不支持的数据集类型: {dataset_type}，尝试使用通用加载方法")
            texts, labels = [], []
        
        if len(texts) == 0:
            logger.warning(f"数据集 {dataset_config.get('name')} 加载失败或无数据")
            return [], [], [], []
        
        logger.info(f"随机划分训练集和测试集 (test_size={test_size})")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_labels = [int(label) for label in train_labels]
        test_labels = [int(label) for label in test_labels]
        
        logger.info(f"数据集划分完成: 训练集 {len(train_texts)} 个样本, 测试集 {len(test_texts)} 个样本")
        return train_texts, train_labels, test_texts, test_labels

# ==================== 主评估函数 ====================
def evaluate_class1(config):
    """评估Class 1: 金融检索任务"""
    model_config = config["model"]
    datasets_config = config["datasets"]["class1"]
    
    logger.info("开始Class 1: 金融信息检索评估")
    logger.info(f"模型: {model_config['name']}")
    logger.info(f"数据集数量: {len(datasets_config)}")
    
    evaluator = FinancialRetrievalEvaluator(model_config["path"])
    
    all_results = []
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"评估数据集: {dataset_name}")
        logger.info(f"类型: {dataset_config['type']}")
        logger.info(f"{'='*60}")
        
        try:
            queries, corpus, qrels = Class1DatasetLoader.load_dataset(dataset_config)
            
            if len(queries) == 0 or len(corpus) == 0:
                logger.warning(f"数据集 {dataset_name} 无数据，跳过")
                continue
            
            logger.info(f"数据统计: {len(queries)} 个查询, {len(corpus)} 个文档")
            
            max_queries = dataset_config.get("sample_size", 3000)
            if len(queries) > max_queries:
                logger.info(f"数据集仍然过大，采样 {max_queries} 个查询进行评估")
                sampled_queries = dict(list(queries.items())[:max_queries])
                sampled_qrels = {qid: qrels[qid] for qid in sampled_queries if qid in qrels}
                queries = sampled_queries
                qrels = sampled_qrels
            
            metrics = evaluator.evaluate_retrieval(
                queries=queries,
                corpus=corpus,
                qrels=qrels,
                top_k_values=[1, 3, 5, 10]
            )
            
            result_record = {
                "dataset": dataset_name,
                "type": dataset_config["type"],
                "description": dataset_config.get("description", ""),
                "score": metrics,
                "stats": {
                    "num_queries": len(queries),
                    "num_documents": len(corpus),
                    "num_qrels": sum(len(v) for v in qrels.values()),
                    "queries_with_qrels": len(qrels)
                }
            }
            
            all_results.append(result_record)
            
            logger.info(f"\n{dataset_name} 评估结果:")
            logger.info(f"查询数量: {len(queries)}")
            logger.info(f"文档数量: {len(corpus)}")
            
            for metric_name, value in metrics.items():
                if "num_queries" not in metric_name and "total_queries" not in metric_name:
                    logger.info(f"{metric_name}: {value:.4f}")
            
        except Exception as e:
            logger.error(f"评估数据集 {dataset_name} 时出错: {e}")
    
    return all_results

def evaluate_class2(config):
    """评估Class 2: 语义文本相似度任务"""
    model_config = config["model"]
    datasets_config = config["datasets"]["class2"]
    
    logger.info("开始Class 2: 语义文本相似度评估")
    logger.info(f"模型: {model_config['name']}")
    logger.info(f"数据集数量: {len(datasets_config)}")
    
    model = STSModel(model_config["path"])
    
    all_results = []
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config["name"]
        file_path = dataset_config["path"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"评估数据集: {dataset_name}")
        logger.info(f"{'='*60}")
        
        try:
            dataset = Class2DatasetLoader.load_dataset(file_path)
            
            if dataset is None:
                logger.warning(f"无法加载数据集 {dataset_name}，跳过")
                continue
            
            if 'data' in dataset:
                split_data = dataset['data']
                
                sentences1 = split_data['sentence1']
                sentences2 = split_data['sentence2']
                scores = split_data['score']
                
                if not isinstance(sentences1, list):
                    sentences1 = list(sentences1)
                    sentences2 = list(sentences2)
                    scores = list(scores)
                
                scores = [float(score) for score in scores]
                
                valid_indices = []
                for i, (s1, s2, score) in enumerate(zip(sentences1, sentences2, scores)):
                    if (isinstance(s1, str) and isinstance(s2, str) and 
                        len(s1.strip()) > 0 and len(s2.strip()) > 0 and
                        not pd.isna(score)):
                        valid_indices.append(i)
                
                if len(valid_indices) < len(sentences1):
                    logger.info(f"过滤掉 {len(sentences1) - len(valid_indices)} 个无效样本")
                    sentences1 = [sentences1[i] for i in valid_indices]
                    sentences2 = [sentences2[i] for i in valid_indices]
                    scores = [scores[i] for i in valid_indices]
                
                logger.info(f"编码句子1...")
                embeddings1 = model.encode(sentences1, batch_size=32)
                
                logger.info(f"编码句子2...")
                embeddings2 = model.encode(sentences2, batch_size=32)
                
                logger.info(f"计算相似度...")
                pred_scores = []
                
                batch_size = 256
                for i in tqdm(range(0, len(embeddings1), batch_size), desc="计算相似度"):
                    batch_emb1 = embeddings1[i:i+batch_size]
                    batch_emb2 = embeddings2[i:i+batch_size]
                    
                    dot_product = np.sum(batch_emb1 * batch_emb2, axis=1)
                    norm1 = np.linalg.norm(batch_emb1, axis=1)
                    norm2 = np.linalg.norm(batch_emb2, axis=1)
                    
                    norm_product = norm1 * norm2
                    norm_product = np.maximum(norm_product, 1e-8)
                    
                    batch_sims = dot_product / norm_product
                    pred_scores.extend(batch_sims.tolist())
                
                # 特殊处理STSB数据集
                if dataset_name == "STSB":
                    # STSB得分范围是0-5，而余弦相似度范围是-1到1
                    # 将余弦相似度(-1到1)映射到0-5的范围
                    logger.info("检测到STSB数据集，将预测分数从[-1,1]映射到[0,5]范围")
                    pred_scores = [(score + 1) * 2.5 for score in pred_scores]
                
                try:
                    pearson_corr, _ = pearsonr(pred_scores, scores)
                    spearman_corr, _ = spearmanr(pred_scores, scores)
                    
                    logger.info(f"评估完成:")
                    logger.info(f"  Pearson相关系数:  {pearson_corr:.4f}")
                    logger.info(f"  Spearman相关系数: {spearman_corr:.4f}")
                    logger.info(f"  得分范围: [{min(scores):.2f}, {max(scores):.2f}]")
                    logger.info(f"  预测得分范围: [{min(pred_scores):.2f}, {max(pred_scores):.2f}]")
                    
                    result = {
                        "dataset": dataset_name,
                        "file_path": file_path,
                        "pearson": float(pearson_corr),
                        "spearman": float(spearman_corr),
                        "num_samples": len(sentences1),
                        "score_range": {
                            "min": float(min(scores)),
                            "max": float(max(scores)),
                            "pred_min": float(min(pred_scores)),
                            "pred_max": float(max(pred_scores))
                        }
                    }
                    
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"计算指标失败: {e}")
                    continue
            else:
                logger.error(f"错误: 数据集中没有找到'data'分割")
                continue
                
        except Exception as e:
            logger.error(f"评估数据集 {dataset_name} 时出错: {e}")
    
    return all_results
    """评估Class 2: 语义文本相似度任务"""
    model_config = config["model"]
    datasets_config = config["datasets"]["class2"]
    
    logger.info("开始Class 2: 语义文本相似度评估")
    logger.info(f"模型: {model_config['name']}")
    logger.info(f"数据集数量: {len(datasets_config)}")
    
    model = STSModel(model_config["path"])
    
    all_results = []
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config["name"]
        file_path = dataset_config["path"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"评估数据集: {dataset_name}")
        logger.info(f"{'='*60}")
        
        try:
            dataset = Class2DatasetLoader.load_dataset(file_path)
            
            if dataset is None:
                logger.warning(f"无法加载数据集 {dataset_name}，跳过")
                continue
            
            if 'data' in dataset:
                split_data = dataset['data']
                
                sentences1 = split_data['sentence1']
                sentences2 = split_data['sentence2']
                scores = split_data['score']
                
                if not isinstance(sentences1, list):
                    sentences1 = list(sentences1)
                    sentences2 = list(sentences2)
                    scores = list(scores)
                
                scores = [float(score) for score in scores]
                
                valid_indices = []
                for i, (s1, s2, score) in enumerate(zip(sentences1, sentences2, scores)):
                    if (isinstance(s1, str) and isinstance(s2, str) and 
                        len(s1.strip()) > 0 and len(s2.strip()) > 0 and
                        not pd.isna(score)):
                        valid_indices.append(i)
                
                if len(valid_indices) < len(sentences1):
                    logger.info(f"过滤掉 {len(sentences1) - len(valid_indices)} 个无效样本")
                    sentences1 = [sentences1[i] for i in valid_indices]
                    sentences2 = [sentences2[i] for i in valid_indices]
                    scores = [scores[i] for i in valid_indices]
                
                logger.info(f"编码句子1...")
                embeddings1 = model.encode(sentences1, batch_size=32)
                
                logger.info(f"编码句子2...")
                embeddings2 = model.encode(sentences2, batch_size=32)
                
                logger.info(f"计算相似度...")
                pred_scores = []
                
                batch_size = 256
                for i in tqdm(range(0, len(embeddings1), batch_size), desc="计算相似度"):
                    batch_emb1 = embeddings1[i:i+batch_size]
                    batch_emb2 = embeddings2[i:i+batch_size]
                    
                    dot_product = np.sum(batch_emb1 * batch_emb2, axis=1)
                    norm1 = np.linalg.norm(batch_emb1, axis=1)
                    norm2 = np.linalg.norm(batch_emb2, axis=1)
                    
                    norm_product = norm1 * norm2
                    norm_product = np.maximum(norm_product, 1e-8)
                    
                    batch_sims = dot_product / norm_product
                    pred_scores.extend(batch_sims.tolist())
                
                try:
                    pearson_corr, _ = pearsonr(pred_scores, scores)
                    spearman_corr, _ = spearmanr(pred_scores, scores)
                    
                    logger.info(f"评估完成:")
                    logger.info(f"  Pearson相关系数:  {pearson_corr:.4f}")
                    logger.info(f"  Spearman相关系数: {spearman_corr:.4f}")
                    
                    result = {
                        "dataset": dataset_name,
                        "file_path": file_path,
                        "pearson": float(pearson_corr),
                        "spearman": float(spearman_corr),
                        "num_samples": len(sentences1)
                    }
                    
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"计算指标失败: {e}")
                    continue
            else:
                logger.error(f"错误: 数据集中没有找到'data'分割")
                continue
                
        except Exception as e:
            logger.error(f"评估数据集 {dataset_name} 时出错: {e}")
    
    return all_results

def evaluate_class3(config):
    """评估Class 3: 文本分类任务"""
    model_config = config["model"]
    datasets_config = config["datasets"]["class3"]
    
    logger.info("开始Class 3: 金融文本分类评估")
    logger.info(f"模型: {model_config['name']}")
    logger.info(f"数据集数量: {len(datasets_config)}")
    
    classifier = FinancialTextClassifier(
        model_path=model_config["path"],
        classifier_type="logistic"  # 默认使用逻辑回归
    )
    
    evaluation_results = []
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"评估数据集: {dataset_name}")
        logger.info(f"类型: {dataset_config.get('type', 'unknown')}")
        logger.info(f"分类器: logistic")
        logger.info(f"{'='*60}")
        
        try:
            train_texts, train_labels, test_texts, test_labels = Class3DatasetLoader.load_dataset(dataset_config)
            
            if len(train_texts) == 0 or len(test_texts) == 0:
                logger.warning(f"数据集 {dataset_name} 无数据，跳过")
                continue
            
            logger.info(f"数据统计: 训练集 {len(train_texts)} 个样本, 测试集 {len(test_texts)} 个样本")
            
            logger.info("编码训练集文本...")
            X_train = classifier.encode_texts(train_texts)
            logger.info("编码测试集文本...")
            X_test = classifier.encode_texts(test_texts)
            
            classifier.train_classifier(X_train, train_labels)
            
            metrics, cls_report = classifier.evaluate_classification(X_test, test_labels)
            
            result = {
                "dataset": dataset_name,
                "type": dataset_config.get("type", "unknown"),
                "description": dataset_config.get("description", ""),
                "score": metrics,
                "sample_count": {
                    "train_samples": len(train_texts),
                    "test_samples": len(test_texts),
                    "num_classes": len(np.unique(train_labels + test_labels))
                }
            }
            
            evaluation_results.append(result)
            
            logger.info(f"\n{dataset_name} 分类评估结果:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
            
        except Exception as e:
            logger.error(f"评估数据集 {dataset_name} 时出错: {e}")
    
    return evaluation_results

def save_results(results, output_dir, class_type, model_config):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        logger.error(f"Class {class_type} 没有有效的评估结果可保存")
        return
    
    model_name = model_config["name"]
    model_name = model_name.replace("\\", "_").replace("/", "_").replace(":", "_")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{model_name}_class{class_type}_{timestamp}.json"
    summary_file = os.path.join(output_dir, output_filename)
    
    output_data = {
        "evaluation_results": results,
        "model_info": {
            "model_name": model_config["name"],
            "model_path": model_config["path"],
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "class_type": class_type,
        "note": f"金融文本评估 Class {class_type}"
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\n汇总结果已保存到: {summary_file}")
    
    return summary_file

def print_summary(results, class_type):
    """打印评估结果摘要"""
    if not results:
        print(f"\nClass {class_type} 没有有效的评估结果")
        return
    
    print(f"\n{'='*100}")
    print(f"金融文本评估汇总 (Class {class_type})")
    print("="*100)
    
    if class_type == "1":
        print(f"{'数据集':<25} {'类型':<20} {'Recall@5':<12} {'Recall@10':<12} {'nDCG@10':<12} {'查询数':<10}")
        print("-"*100)
        
        for result in results:
            metrics = result["score"]
            dataset_name = result["dataset"]
            dataset_type = result.get("type", "unknown")
            recall_5 = metrics.get("Recall@5", 0)
            recall_10 = metrics.get("Recall@10", 0)
            ndcg_10 = metrics.get("nDCG@10", 0)
            num_queries = result["stats"]["num_queries"]
            
            print(f"{dataset_name:<25} {dataset_type:<20} {recall_5:<12.4f} {recall_10:<12.4f} {ndcg_10:<12.4f} {num_queries:<10}")
        
        print("="*100)
        
        if results:
            recall_5_avg = np.mean([r["score"].get("Recall@5", 0) for r in results])
            recall_10_avg = np.mean([r["score"].get("Recall@10", 0) for r in results])
            ndcg_10_avg = np.mean([r["score"].get("nDCG@10", 0) for r in results])
            
            print(f"\n平均指标:")
            print(f"  平均Recall@5:  {recall_5_avg:.4f}")
            print(f"  平均Recall@10: {recall_10_avg:.4f}")
            print(f"  平均nDCG@10:   {ndcg_10_avg:.4f}")
            print(f"  评估的数据集数: {len(results)}")
    
    elif class_type == "2":
        print(f"{'数据集':<25} {'Pearson':<12} {'Spearman':<12} {'样本数':<10} {'文件类型':<10}")
        print("-"*100)
        
        for result in results:
            dataset_name = result["dataset"]
            pearson = result["pearson"]
            spearman = result["spearman"]
            num_samples = result["num_samples"]
            file_path = result.get("file_path", "")
            file_ext = os.path.splitext(file_path)[1] if file_path else "N/A"
            
            print(f"{dataset_name:<25} {pearson:<12.4f} {spearman:<12.4f} {num_samples:<10} {file_ext:<10}")
        
        print("="*100)
        
        if results:
            avg_pearson = np.mean([r["pearson"] for r in results])
            avg_spearman = np.mean([r["spearman"] for r in results])
            total_samples = sum([r["num_samples"] for r in results])
            
            print(f"\n平均值统计:")
            print(f"  平均Pearson相关系数:  {avg_pearson:.4f}")
            print(f"  平均Spearman相关系数: {avg_spearman:.4f}")
            print(f"  总样本数: {total_samples}")
            print(f"  评估的数据集数: {len(results)}")
    
    elif class_type == "3":
        print(f"{'数据集':<40} {'F1_macro':<12} {'F1_micro':<12} {'Accuracy':<12} {'MAP':<12}")
        print("-"*100)
        
        for result in results:
            dataset_name = result["dataset"]
            score = result["score"]
            
            f1_macro = score.get("F1_macro", 0)
            f1_micro = score.get("F1_micro", 0)
            accuracy = score.get("Accuracy", 0)
            map_score = score.get("Mean_Average_Precision", score.get("Average_Precision", 0))
            
            print(f"{dataset_name:<40} {f1_macro:<12.4f} {f1_micro:<12.4f} {accuracy:<12.4f} {map_score:<12.4f}")
        
        print("="*100)
        
        if results:
            avg_f1_macro = np.mean([r["score"].get("F1_macro", 0) for r in results])
            avg_f1_micro = np.mean([r["score"].get("F1_micro", 0) for r in results])
            avg_accuracy = np.mean([r["score"].get("Accuracy", 0) for r in results])
            avg_map = np.mean([r["score"].get("Mean_Average_Precision", r["score"].get("Average_Precision", 0)) for r in results])
            
            print(f"\n平均指标统计:")
            print(f"  平均F1_macro:  {avg_f1_macro:.4f}")
            print(f"  平均F1_micro:  {avg_f1_micro:.4f}")
            print(f"  平均Accuracy:  {avg_accuracy:.4f}")
            print(f"  平均MAP:       {avg_map:.4f}")
            print(f"  评估的数据集数: {len(results)}")

# ==================== 主程序 ====================
def main():
    """主程序入口"""
    print("="*70)
    print("金融文本评估系统")
    print("="*70)
    print("1: 金融信息检索评估 (Class 1)")
    print("2: 语义文本相似度评估 (Class 2)")
    print("3: 金融文本分类评估 (Class 3)")
    print("a: 评估所有类别")
    print("="*70)
    
    choice = input("请选择要评估的类别 (1/2/3/a): ").strip().lower()
    
    if choice not in ["1", "2", "3", "a"]:
        print("无效选择，请重新运行程序")
        return
    
    # 加载配置文件
    config = load_config()
    model_config = config["model"]
    
    # 创建输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "results")
    
    # 根据选择执行评估
    if choice == "1" or choice == "a":
        print("\n" + "="*70)
        print("开始Class 1: 金融信息检索评估")
        print("="*70)
        results = evaluate_class1(config)
        save_results(results, output_dir, "1", model_config)
        print_summary(results, "1")
    
    if choice == "2" or choice == "a":
        print("\n" + "="*70)
        print("开始Class 2: 语义文本相似度评估")
        print("="*70)
        results = evaluate_class2(config)
        save_results(results, output_dir, "2", model_config)
        print_summary(results, "2")
    
    if choice == "3" or choice == "a":
        print("\n" + "="*70)
        print("开始Class 3: 金融文本分类评估")
        print("="*70)
        results = evaluate_class3(config)
        save_results(results, output_dir, "3", model_config)
        print_summary(results, "3")
    
    print("\n评估完成！")

if __name__ == "__main__":
    main()