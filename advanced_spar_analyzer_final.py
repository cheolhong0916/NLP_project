#!/usr/bin/env python3
"""
SPAR-Bench 간소화된 통합 분석기 (핵심 5개 메트릭)
가장 중요한 text similarity 메트릭들만 사용하여 효율적이고 안정적인 평가

핵심 메트릭:
1. Exact Match - 기준선
2. BLEU Score - 전통적 NLP 메트릭
3. Embedding Similarity - 의미적 유사성 
4. Spatial Similarity - 도메인 특화
5. Combined Similarity - 통합 점수

의존성 설치:
    pip install nltk sentence-transformers scikit-learn

사용법:
    python simplified_spar_analyzer.py                           # 기본 파일로 완전 분석
    python simplified_spar_analyzer.py my_file.jsonl            # 지정 파일로 완전 분석  
    python simplified_spar_analyzer.py --quick                  # 빠른 성능 체크
    python simplified_spar_analyzer.py --quick my_file.jsonl    # 지정 파일로 빠른 체크
"""

import json
import pandas as pd
import numpy as np
import re
import math
import sys
import os
import warnings
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

# 선택적 라이브러리 import
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️  NLTK not available. BLEU scores will be disabled.")
    NLTK_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not available. Embedding similarity will be disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# 경고 무시 (RobertaModel 경고 포함)
warnings.filterwarnings("ignore")

# ============================================================================
# JSON 직렬화 지원 함수들
# ============================================================================

def convert_to_serializable(obj):
    """numpy 타입들을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def safe_json_dump(data, file_path):
    """안전한 JSON 저장"""
    serializable_data = convert_to_serializable(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

# ============================================================================
# 전역 모델 초기화 (lazy loading)
# ============================================================================

_sentence_model = None

def get_sentence_model():
    """Sentence Transformer 모델을 lazy loading으로 가져오기"""
    global _sentence_model
    if _sentence_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        # print("🔄 Loading sentence transformer model...")
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        # print("✅ Sentence transformer loaded!")
    return _sentence_model

# ============================================================================

def exact_match_score(pred: str, target: str) -> float:
    """Exact Match - 완전 일치 기준선"""
    pred_clean = pred.lower().strip()
    target_clean = target.lower().strip()
    return 1.0 if pred_clean == target_clean else 0.0

def bleu_score(pred: str, target: str) -> float:
    """BLEU Score - 전통적 NLP 메트릭"""
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        pred_tokens = pred.lower().split()
        target_tokens = target.lower().split()
        
        if not pred_tokens or not target_tokens:
            return 0.0
        
        # BLEU score 계산 (smoothing 적용)
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)
        return float(score)
    except:
        return 0.0

def embedding_similarity_score(pred: str, target: str) -> float:
    """Embedding Similarity - 의미적 유사성"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return 0.0
    
    try:
        model = get_sentence_model()
        if model is None:
            return 0.0
        
        # 임베딩 계산
        pred_emb = model.encode([pred])
        target_emb = model.encode([target])
        
        # 코사인 유사도
        similarity = cosine_similarity(pred_emb, target_emb)[0][0]
        return float(similarity)
    except:
        return 0.0

def spatial_similarity_score(pred: str, target: str) -> float:
    """Spatial Similarity - 공간 관계 특화"""
    # 확장된 공간 관계 용어 매핑
    spatial_synonyms = {
        'left': ['left', 'leftward', 'to the left', 'on the left', 'left side', 'left of'],
        'right': ['right', 'rightward', 'to the right', 'on the right', 'right side', 'right of'],
        'above': ['above', 'over', 'on top', 'higher', 'up', 'top of', 'upper', 'overhead'],
        'below': ['below', 'under', 'beneath', 'lower', 'down', 'bottom of', 'underneath'],
        'front': ['front', 'forward', 'ahead', 'in front', 'in front of', 'before'],
        'back': ['back', 'behind', 'backward', 'rear', 'back of'],
        'near': ['near', 'close', 'nearby', 'adjacent', 'closer', 'next to'],
        'far': ['far', 'distant', 'away', 'remote', 'farther', 'far from'],
        'inside': ['inside', 'within', 'in', 'interior', 'inner'],
        'outside': ['outside', 'out', 'external', 'exterior', 'outer'],
        'center': ['center', 'middle', 'central', 'midst'],
        'corner': ['corner', 'edge', 'side']
    }
    
    pred_lower = pred.lower()
    target_lower = target.lower()

    if pred_lower in target_lower or target_lower in pred_lower:
        return 1.0

    pred_concepts = set()
    target_concepts = set()
    
    for concept, synonyms in spatial_synonyms.items():
        pred_has = any(syn in pred_lower for syn in synonyms)
        target_has = any(syn in target_lower for syn in synonyms)
        
        if pred_has:
            pred_concepts.add(concept)
        if target_has:
            target_concepts.add(concept)

    if not target_concepts:
        return 0.0
    
    if pred_concepts & target_concepts:
        overlap_ratio = len(pred_concepts & target_concepts) / len(target_concepts)
        return float(0.9 * overlap_ratio)
    
    return 0.0

def rouge_score(pred: str, target: str) -> float:
    """ROUGE Score - Recall 기반 메트릭"""
    try:
        pred_tokens = set(pred.lower().split())
        target_tokens = set(target.lower().split())
        
        if not target_tokens:
            return 0.0
        
        # ROUGE-1: unigram recall
        overlap = len(pred_tokens & target_tokens)
        rouge_1_recall = overlap / len(target_tokens)
        
        return float(rouge_1_recall)
    except:
        return 0.0

def combined_similarity_score(pred: str, target: str) -> float:
    """Combined Similarity - 통합 점수 (exact_match 제외)"""
    # 각 메트릭 계산
    bleu = bleu_score(pred, target)
    rouge = rouge_score(pred, target)
    embedding = embedding_similarity_score(pred, target)
    spatial = spatial_similarity_score(pred, target)
    
    # 가중 평균 계산
    scores = []
    weights = []
    
    # 사용 가능한 메트릭들로 가중 평균 (exact_match 제외)
    if embedding > 0:
        scores.append(embedding)
        weights.append(0.5)  # 의미적 유사성 50%
    
    if spatial > 0:
        scores.append(spatial)
        weights.append(0.3)  # 공간 관계 30%
    
    if bleu > 0:
        scores.append(bleu)
        weights.append(0.1)  # BLEU 10%
    
    if rouge > 0:
        scores.append(rouge)
        weights.append(0.1)  # ROUGE 10%
    
    if scores:
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
            combined = sum(s*w for s, w in zip(scores, weights))
            return float(combined)
    
    return 0.0 


def comprehensive_similarity_evaluation(pred: str, target: str) -> Dict[str, float]:
    """포괄적인 메트릭 평가 (exact_match 제외)"""
    results = {
        'bleu': bleu_score(pred, target),
        'rouge': rouge_score(pred, target),
        'embedding_similarity': embedding_similarity_score(pred, target),
        'spatial_similarity': spatial_similarity_score(pred, target),
    }
    
    # Combined similarity는 다른 메트릭들을 기반으로 계산
    results['combined_similarity'] = combined_similarity_score(pred, target)
    
    return results

# ============================================================================
# SPAR 벤치마크 평가 함수들
# ============================================================================

MCA_QUESTION_TYPES = [
    "obj_spatial_relation_oo", "obj_spatial_relation_oc_mv",
    "spatial_imagination_oc", "spatial_imagination_oo", "spatial_imagination_oc_mv", "spatial_imagination_oo_mv",
    "position_matching", "camera_motion_infer", "distance_infer_center_oo", "distance_infer_center_oo_mv"
]

SENTENCE_QUESTION_TYPES = ["obj_spatial_relation_oo_mv"]

NA_QUESTION_TYPES = [
    "depth_prediction_oc", "depth_prediction_oo", "distance_prediction_oc", "distance_prediction_oo",
    "depth_prediction_oc_mv", "depth_prediction_oo_mv", "distance_prediction_oo_mv", "distance_prediction_oc_mv",  
]

SPECIAL_QUESTION_TYPES = ["view_change_infer"]

Low = [
    "depth_prediction_oc", "depth_prediction_oo", "distance_prediction_oc", "distance_prediction_oo",
    "depth_prediction_oc_mv", "depth_prediction_oo_mv", "distance_prediction_oo_mv", "distance_prediction_oc_mv",  
]

Middle = ["view_change_infer", "position_matching", "camera_motion_infer"]

High = [
    "obj_spatial_relation_oo", "obj_spatial_relation_oc_mv", "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc", "spatial_imagination_oo", "spatial_imagination_oc_mv", 
    "spatial_imagination_oo_mv", "distance_infer_center_oo", "distance_infer_center_oo_mv"
]

def extract_task_from_prompt(prompt: str) -> str:
    """Task type is fixed as obj_spatial_relation_oo_mv"""
    return "obj_spatial_relation_oo_mv"

def process_na_prediction(pred, task):
    """수치 답변에서 숫자 추출"""
    numbers = re.findall(r'(?<!\^)\d+\.\d+|(?<!\^)\d+', pred)
    extracted_numbers = [float(num) if '.' in num else int(num) for num in numbers]
    
    if task in ["depth_prediction_oc_mv", "depth_prediction_oo_mv", 
                "distance_prediction_oc_mv", "distance_prediction_oo_mv"]:
        if len(extracted_numbers) == 0:
            extracted_numbers = [-1]
        extracted_numbers = [extracted_numbers[-1]]
    
    return extracted_numbers[0] if extracted_numbers else 0

def abs_dist_norm(pred, target):
    """절대 정규화된 거리"""
    if target == 0.0:
        return abs(pred - target)
    else:
        return abs((pred - target) / target)

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    """평균 상대 정확도"""
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return float(accuracy.mean())

def compute_vci_metric(pred, answer):
    """view_change_infer 메트릭 계산 (간단 버전)"""
    try:
        return 1.0 if pred.lower().strip() == answer.lower().strip() else 0.0
    except:
        return 0.0

# ============================================================================
# SPAR 평가기 클래스들
# ============================================================================

class SPARBenchEvaluator:
    """간소화된 SPAR 벤치마크 평가기 (핵심 5개 메트릭)"""
    
    def __init__(self):
        # print("🚀 Initializing Simplified SPAR Evaluator (5 Core Metrics)...")
        self._print_available_metrics()
    
    def _print_available_metrics(self):
        """사용 가능한 메트릭 출력"""
        # print("📊 Core 5 similarity metrics:")
        # print(f"   ✅ 1. Exact Match: Always available")
        # print(f"   {'✅' if NLTK_AVAILABLE else '❌'} 2. BLEU Score: {'Available' if NLTK_AVAILABLE else 'Disabled'}")
        # print(f"   {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌'} 3. Embedding Similarity: {'Available' if SENTENCE_TRANSFORMERS_AVAILABLE else 'Disabled'}")
        # print(f"   ✅ 4. Spatial Similarity: Always available")
        # print(f"   ✅ 5. Combined Similarity: Always available")
    
    def load_jsonl_data(self, jsonl_file: str) -> List[Dict]:
        """JSONL 파일에서 모든 데이터 로드"""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        task = extract_task_from_prompt(sample['prompt'])
                        sample['task'] = task
                        sample['sample_id'] = line_num
                        data.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        
        # print(f"✅ Loaded {len(data)} samples from {jsonl_file}")
        return data
    
    def process_single_result(self, sample: Dict) -> Dict:
        """단일 샘플에 대한 간소화된 평가 수행"""
        task = sample['task']
        prediction = sample['predict']
        answer = sample['label']
        
        result = {'prediction': prediction}
        
        if task in MCA_QUESTION_TYPES:
            # Multiple Choice Questions (객관식)
            result['accuracy'] = exact_match_score(prediction, answer)
            result['answer_format'] = 'select'
            
        elif task in SENTENCE_QUESTION_TYPES:
            # Sentence Questions - 핵심 메트릭 평가
            similarity_results = comprehensive_similarity_evaluation(prediction, answer)
            
            # 모든 메트릭 저장
            result.update(similarity_results)
            result['accuracy'] = similarity_results['combined_similarity']
            result['answer_format'] = 'sentence'
            
        elif task in NA_QUESTION_TYPES:
            # Numerical Answer Questions
            try:
                pred_val = process_na_prediction(prediction, task)
                target_val = float(answer)
                result['MRA:.5:.95:.05'] = mean_relative_accuracy(pred_val, target_val)
            except:
                result['MRA:.5:.95:.05'] = 0.0
                
        elif task in SPECIAL_QUESTION_TYPES:
            # Special Questions
            if task == "view_change_infer":
                result['vci_metric'] = compute_vci_metric(prediction, answer)
        
        return result
    
    def evaluate(self, jsonl_file: str) -> Dict[str, float]:
        """JSONL 파일만으로 평가 수행"""
        data = self.load_jsonl_data(jsonl_file)
        
        processed_results = []
        for sample in data:
            result = self.process_single_result(sample)
            result.update({
                'task': sample['task'],
                'sample_id': sample['sample_id']
            })
            processed_results.append(result)
        
        return self.aggregate_results(processed_results)
    
    def aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """결과 집계"""
        df = pd.DataFrame(results)
        output = {}
        
        # 태스크별 결과 계산
        for task_type, task_indexes in df.groupby('task').groups.items():
            per_task_type = df.iloc[task_indexes]
            
            if task_type in MCA_QUESTION_TYPES:
                output[f"{task_type}_accuracy"] = float(per_task_type['accuracy'].mean())
                
            elif task_type in SENTENCE_QUESTION_TYPES:
                # Sentence 형식 - 핵심 메트릭 집계
                core_metrics = ['embedding_similarity', 'spatial_similarity', 'bleu', 'rouge', 'combined_similarity']
                
                for metric in core_metrics:
                    if metric in per_task_type.columns and not per_task_type[metric].isna().all():
                        output[f"{task_type}_{metric}"] = float(per_task_type[metric].mean())
                
            elif task_type in NA_QUESTION_TYPES:
                output[f"{task_type}_MRA:.5:.95:.05"] = float(per_task_type['MRA:.5:.95:.05'].mean())
                
            elif task_type in SPECIAL_QUESTION_TYPES:
                if task_type == "view_change_infer":
                    output[f"{task_type}_vci_metric"] = float(per_task_type['vci_metric'].mean())
        
        # 전체 점수 계산 (주요 메트릭만 사용)
        main_scores = []
        for k, v in output.items():
            if k.endswith('_accuracy') or k.endswith('_combined_similarity') or k.endswith('_MRA:.5:.95:.05') or k.endswith('_vci_metric'):
                main_scores.append(v)
        
        if main_scores:
            output['overall'] = float(sum(main_scores) / len(main_scores))
        
        # 레벨별 점수 계산
        low_scores = [v for k, v in output.items() 
                     if any(task in k for task in Low) and ('_accuracy' in k or '_MRA' in k or '_combined_similarity' in k)]
        middle_scores = [v for k, v in output.items() 
                        if any(task in k for task in Middle) and ('_accuracy' in k or '_vci' in k)]
        high_scores = [v for k, v in output.items() 
                      if any(task in k for task in High) and ('_accuracy' in k or '_combined_similarity' in k)]
        
        if low_scores:
            output['Low'] = float(np.mean(low_scores))
        if middle_scores:
            output['Middle'] = float(np.mean(middle_scores))
        if high_scores:
            output['High'] = float(np.mean(high_scores))
        
        return output

class DetailedSPARAnalyzer(SPARBenchEvaluator):
    """상세 SPAR 벤치마크 분석기 (간소화된 버전)"""
    
    def __init__(self):
        super().__init__()
        self.detailed_results = []
        
    def detailed_evaluate(self, jsonl_file: str, save_results: bool = True):
        """상세 평가 수행 및 결과 저장"""
        # print("🔍 Starting Simplified SPAR Evaluation (5 Core Metrics)...")
        
        data = self.load_jsonl_data(jsonl_file)
        # print("📊 Processing each sample with core similarity metrics...")
        
        detailed_results = []
        for i, sample in enumerate(data):
            # if i % 50 == 0:
            #     print(f"   Progress: {i}/{len(data)} samples processed...")
                
            result = self.process_single_result(sample)
            
            detailed_info = {
                'sample_id': sample['sample_id'],
                'task': sample['task'],
                'prompt_snippet': sample['prompt'][:200] + "...",
                'prediction': sample['predict'],
                'ground_truth': sample['label'],
                'prediction_length': len(sample['predict']),
                'ground_truth_length': len(sample['label']),
                **result
            }
            
            detailed_results.append(detailed_info)
        
        self.detailed_results = detailed_results
        
        aggregated_results = self.aggregate_results(detailed_results)
        
        # if save_results:
        #     self.save_detailed_results("spar_simplified_evaluation.json")
            
        # print("✅ Simplified evaluation completed!")
        return aggregated_results, detailed_results
    
    def save_detailed_results(self, output_file: str):
        """상세 결과를 JSON 파일로 안전하게 저장"""
        results_to_save = {
            'summary': self.get_evaluation_summary(),
            'core_metrics_analysis': self.analyze_core_metrics(),
            'task_breakdown': self.get_task_breakdown(),
            'detailed_samples': self.detailed_results[:100],  # 처음 100개만 저장 (크기 제한)
            'failure_analysis': self.analyze_failures(),
            'performance_stats': self.get_performance_stats()
        }
        
        # 안전한 JSON 저장
        # safe_json_dump(results_to_save, output_file)
        # print(f"💾 Simplified results saved to: {output_file}")
    
    def analyze_core_metrics(self):
        """핵심 5개 메트릭 상세 분석"""
        if not self.detailed_results:
            return {}
            
        df = pd.DataFrame(self.detailed_results)
        sentence_tasks = df[df['task'].isin(SENTENCE_QUESTION_TYPES)]
        
        if sentence_tasks.empty:
            return {}
        
        analysis = {}
        
        # 핵심 5개 메트릭
        core_metrics = ['embedding_similarity', 'spatial_similarity', 'bleu', 'rouge', 'combined_similarity']
        
        for metric in core_metrics:
            if metric in sentence_tasks.columns and not sentence_tasks[metric].isna().all():
                analysis[metric] = {
                    'mean': float(sentence_tasks[metric].mean()),
                    'std': float(sentence_tasks[metric].std()),
                    'median': float(sentence_tasks[metric].median()),
                    'min': float(sentence_tasks[metric].min()),
                    'max': float(sentence_tasks[metric].max()),
                    'high_performance_count': int((sentence_tasks[metric] >= 0.8).sum()),
                    'high_performance_rate': float((sentence_tasks[metric] >= 0.8).mean()),
                    'perfect_count': int((sentence_tasks[metric] >= 0.99).sum()),
                    'perfect_rate': float((sentence_tasks[metric] >= 0.99).mean())
                }
        
        return analysis
    
    def get_evaluation_summary(self):
        """평가 요약 정보"""
        if not self.detailed_results:
            return {}
            
        df = pd.DataFrame(self.detailed_results)
        
        summary = {
            'total_samples': int(len(df)),
            'unique_tasks': int(df['task'].nunique()),
            'task_distribution': {k: int(v) for k, v in df['task'].value_counts().to_dict().items()},
            'answer_format_distribution': {k: int(v) for k, v in df['answer_format'].value_counts().to_dict().items()} if 'answer_format' in df.columns else {}
        }
        
        # 핵심 메트릭별 전체 평균
        metrics = {}
        core_metric_cols = ['embedding_similarity', 'spatial_similarity', 'bleu', 'rouge', 'combined_similarity']

        for metric in core_metric_cols:
            if metric in df.columns and not df[metric].isna().all():
                metrics[f'avg_{metric}'] = float(df[metric].mean())
        
        summary['metrics'] = metrics
        return summary
    
    def get_task_breakdown(self):
        """태스크별 상세 분석"""
        if not self.detailed_results:
            return {}
            
        df = pd.DataFrame(self.detailed_results)
        task_breakdown = {}
        
        for task in df['task'].unique():
            task_data = df[df['task'] == task]
            
            breakdown = {
                'sample_count': int(len(task_data)),
                'task_type': self._get_task_type(task),
                'answer_format': task_data['answer_format'].iloc[0] if 'answer_format' in task_data.columns else 'unknown'
            }
            
            # 핵심 메트릭들에 대한 통계
            core_metrics = ['accuracy', 'exact_match', 'bleu', 'embedding_similarity', 'spatial_similarity', 'combined_similarity', 'MRA:.5:.95:.05', 'vci_metric']
            
            for metric in core_metrics:
                if metric in task_data.columns and not task_data[metric].isna().all():
                    breakdown[metric] = {
                        'mean': float(task_data[metric].mean()),
                        'std': float(task_data[metric].std()),
                        'median': float(task_data[metric].median()),
                        'high_count': int((task_data[metric] >= 0.8).sum()),
                        'total_count': int(len(task_data))
                    }
            
            task_breakdown[task] = breakdown
        
        return task_breakdown
    
    def _get_task_type(self, task):
        """태스크 타입 분류"""
        if task in MCA_QUESTION_TYPES:
            return "Multiple Choice (MCA)"
        elif task in SENTENCE_QUESTION_TYPES:
            return "Simplified Sentence Format (5 Metrics)"
        elif task in NA_QUESTION_TYPES:
            return "Numerical Answer (NA)"
        elif task in SPECIAL_QUESTION_TYPES:
            return "Special"
        else:
            return "Unknown"
    
    def analyze_failures(self):
        """실패 케이스 분석"""
        if not self.detailed_results:
            return {}
            
        df = pd.DataFrame(self.detailed_results)
        failures = []
        
        for _, row in df.iterrows():
            is_failure = False
            failure_reasons = []
            
            # 통합 점수 기반 실패 판정
            if 'combined_similarity' in row and pd.notna(row['combined_similarity']) and row['combined_similarity'] < 0.3:
                is_failure = True
                failure_reasons.append("Low combined similarity")
            elif 'accuracy' in row and pd.notna(row['accuracy']) and row['accuracy'] < 0.3:
                is_failure = True
                failure_reasons.append("Low accuracy")
            
            # Sentence 형식 추가 체크
            if row['task'] in SENTENCE_QUESTION_TYPES:
                low_metrics = []
                if 'exact_match' in row and pd.notna(row['exact_match']) and row['exact_match'] == 0:
                    low_metrics.append('exact_match')
                if 'embedding_similarity' in row and pd.notna(row['embedding_similarity']) and row['embedding_similarity'] < 0.3:
                    low_metrics.append('embedding_similarity')
                if 'spatial_similarity' in row and pd.notna(row['spatial_similarity']) and row['spatial_similarity'] < 0.3:
                    low_metrics.append('spatial_similarity')
                
                if len(low_metrics) >= 2:
                    is_failure = True
                    failure_reasons.append(f"Multiple low metrics: {', '.join(low_metrics)}")
            
            if is_failure:
                failure_info = {
                    'sample_id': int(row['sample_id']),
                    'task': row['task'],
                    'prediction': row['prediction'][:150] + "..." if len(row['prediction']) > 150 else row['prediction'],
                    'ground_truth': row['ground_truth'][:150] + "..." if len(row['ground_truth']) > 150 else row['ground_truth'],
                    'failure_reasons': failure_reasons,
                    'metrics': {}
                }
                
                # 핵심 메트릭 추가
                important_metrics = ['accuracy', 'combined_similarity', 'exact_match', 'embedding_similarity', 'spatial_similarity']
                for metric in important_metrics:
                    if metric in row:
                        val = row[metric]
                        failure_info['metrics'][metric] = float(val) if pd.notna(val) else None
                
                failures.append(failure_info)
        
        return {
            'total_failures': len(failures),
            'failure_rate': float(len(failures) / len(df)) if len(df) > 0 else 0.0,
            'failures_by_task': {k: int(v) for k, v in Counter([f['task'] for f in failures]).items()},
            'sample_failures': failures[:10]  # 상위 10개만 저장
        }
    
    def get_performance_stats(self):
        """성능 통계"""
        if not self.detailed_results:
            return {}
            
        df = pd.DataFrame(self.detailed_results)
        stats = {
            'prediction_length': {
                'mean': float(df['prediction_length'].mean()),
                'median': float(df['prediction_length'].median()),
                'std': float(df['prediction_length'].std()),
                'min': int(df['prediction_length'].min()),
                'max': int(df['prediction_length'].max())
            },
            'ground_truth_length': {
                'mean': float(df['ground_truth_length'].mean()),
                'median': float(df['ground_truth_length'].median()),
                'std': float(df['ground_truth_length'].std())
            }
        }
        
        # 길이와 성능의 관계
        performance_cols = ['accuracy', 'combined_similarity', 'embedding_similarity']
        length_correlations = {}
        
        for perf_col in performance_cols:
            if perf_col in df.columns and not df[perf_col].isna().all():
                pred_length_corr = df['prediction_length'].corr(df[perf_col])
                gt_length_corr = df['ground_truth_length'].corr(df[perf_col])
                
                if not np.isnan(pred_length_corr):
                    length_correlations[f'pred_length_vs_{perf_col}'] = float(pred_length_corr)
                if not np.isnan(gt_length_corr):
                    length_correlations[f'gt_length_vs_{perf_col}'] = float(gt_length_corr)
        
        stats['length_performance_correlations'] = length_correlations
        return stats
    
    def print_evaluation_report(self):
        """간소화된 평가 보고서 출력"""
        if not self.detailed_results:
            print("❌ No evaluation results available. Run detailed_evaluate() first.")
            return

        
        # 1. 기본 요약
        summary = self.get_evaluation_summary()
        
        # 2. 핵심 메트릭 요약
        metrics = summary.get('metrics', {})
        if metrics:
            print(f"\n📊 Average Performance (5 Core Metrics):")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
        
    
    def export_csv_results(self, output_file: str = "spar_simplified_results.csv"):
        """결과를 CSV로 내보내기"""
        if not self.detailed_results:
            print("❌ No results to export")
            return
        
        df = pd.DataFrame(self.detailed_results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        # print(f"📄 Simplified results exported to: {output_file}")
    
    def show_metric_comparison(self, n: int = 10):
        """핵심 5개 메트릭 성능 비교"""
        if not self.detailed_results:
            print("❌ No results available")
            return
        
        df = pd.DataFrame(self.detailed_results)
        sentence_tasks = df[df['task'].isin(SENTENCE_QUESTION_TYPES)]
        
        if sentence_tasks.empty:
            print("❌ No sentence tasks found")
            return


def print_simplified_metric_documentation():
    pass

    
    print("""
🎯 1. EXACT MATCH
   Description: Perfect string matching (case-insensitive)
   Range: [0.0, 1.0]
   Use: Strict baseline evaluation

📊 2. BLEU SCORE  
   Description: N-gram overlap precision (from machine translation)
   Range: [0.0, 1.0]
   Use: Fluency and word order assessment

🧠 3. EMBEDDING SIMILARITY
   Description: Semantic similarity using sentence embeddings
   Model: all-MiniLM-L6-v2
   Range: [0.0, 1.0]
   Use: Captures meaning beyond word overlap

🗺️ 4. SPATIAL SIMILARITY
   Description: Domain-specific spatial relationship matching
   Range: [0.0, 1.0]
   Use: Spatial reasoning evaluation

🎪 5. COMBINED SIMILARITY
   Description: Weighted average of all metrics
   Weights: Embedding(40%) + Spatial(30%) + BLEU(20%) + Token(10%)
   Range: [0.0, 1.0]
   Use: Single comprehensive score

📈 Performance Levels:
   🏆 Excellent: ≥ 0.8
   ✅ Good: 0.6 - 0.8
   ⚠️  Fair: 0.4 - 0.6
   ❌ Poor: < 0.4
""")

# ============================================================================
# 실행 함수들
# ============================================================================

def run_complete_analysis(jsonl_file: str):
    """완전한 간소화된 분석 실행"""
    analyzer = DetailedSPARAnalyzer()
    
    # print("🚀 Starting Complete Simplified SPAR Analysis (5 Core Metrics)...")
    
    # 1. 상세 평가 실행
    aggregated, detailed = analyzer.detailed_evaluate(jsonl_file)
    
    # 2. 보고서 출력
    analyzer.print_evaluation_report()
    
    # 3. CSV 내보내기
    analyzer.export_csv_results()
    
    # 4. 메트릭 비교
    analyzer.show_metric_comparison(10)
    
    # print("\n✅ Complete simplified analysis finished!")
    # print("📁 Files generated:")
    # print("   - spar_simplified_evaluation.json (complete results with 5 core metrics)")
    # print("   - spar_simplified_results.csv (spreadsheet format)")
    
    return analyzer

def quick_check(jsonl_file: str):
    """빠른 성능 체크 (간소화된 버전)"""
    print("⚡ Quick Simplified Performance Check (5 Core Metrics)")
    print("-" * 60)
    
    evaluator = SPARBenchEvaluator()
    results = evaluator.evaluate(jsonl_file)
    
    print(f"Overall Score: {results.get('overall', 0):.4f}")
    print(f"Low-level: {results.get('Low', 0):.4f}")
    print(f"Mid-level: {results.get('Middle', 0):.4f}")  
    print(f"High-level: {results.get('High', 0):.4f}")
    
    # obj_spatial_relation_oo_mv 핵심 메트릭
    task_name = "obj_spatial_relation_oo_mv"
    print(f"\n{task_name} Core 5 Metrics:")
    # core_metrics = ['accuracy', 'exact_match', 'bleu', 'embedding_similarity', 'spatial_similarity', 'combined_similarity']
    core_metrics = ['embedding_similarity', 'spatial_similarity', 'bleu', 'rouge', 'combined_similarity']
    
    for key, value in results.items():
        if task_name in key:
            metric_name = key.replace(f"{task_name}_", "")
            if metric_name in core_metrics:
                print(f"  {metric_name}: {value:.4f}")








def extract_model_name_from_folder(folder_name: str) -> str:
    """폴더명에서 모델명 추출"""
    # eval_2025-06-01-qwen-both -> qwen-both
    # eval_2025-06-01-Both -> Both
    parts = folder_name.split('-')
    if len(parts) >= 4:  # eval-2025-06-01-model-name
        return '-'.join(parts[3:])  # model-name 부분만 추출
    elif len(parts) == 3:  # eval-2025-06-01-Model
        return parts[2]  # Model 부분만 추출
    else:
        return folder_name

def scan_eval_folders(base_dir: str = "Eval_Results") -> List[Tuple[str, str]]:
    """평가 폴더들을 스캔하여 (모델명, jsonl_파일_경로) 리스트 반환"""
    if not os.path.exists(base_dir):
        print(f"❌ Error: Directory not found: {base_dir}")
        return []
    
    valid_folders = []
    
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # 폴더가 아니면 스킵
        if not os.path.isdir(folder_path):
            continue
            
        # "None"이 들어간 폴더는 제외
        if "none" in folder_name.lower():
            print(f"⏭️  Skipping folder with 'None': {folder_name}")
            continue
            
        # generated_predictions.jsonl 파일 찾기
        jsonl_path = os.path.join(folder_path, "generated_predictions.jsonl")
        if os.path.exists(jsonl_path):
            model_name = extract_model_name_from_folder(folder_name)
            valid_folders.append((model_name, jsonl_path))
            print(f"✅ Found: {model_name} -> {jsonl_path}")
        else:
            print(f"⚠️  No generated_predictions.jsonl found in: {folder_name}")
    
    return valid_folders

def run_batch_evaluation(base_dir: str = "Eval_Results"):
    """여러 폴더에 대해 일괄 평가 수행"""
    print("🚀 Starting Batch SPAR Evaluation")
    # print("=" * 60)
    
    # 평가 폴더들 스캔
    valid_folders = scan_eval_folders(base_dir)
    
    if not valid_folders:
        print("❌ No valid evaluation folders found!")
        return

    
    # 각 폴더에 대해 평가 수행
    all_results = {}
    
    for model_name, jsonl_path in valid_folders:
        print(f"\n🔄 Evaluating: {model_name}")
        try:
            evaluator = SPARBenchEvaluator()
            results = evaluator.evaluate(jsonl_path)
            all_results[model_name] = results
            print(f"✅ Completed: {model_name}")
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            all_results[model_name] = None
    

    for model_name, results in all_results.items():
        if results is None:
            print(f"\n{model_name}")
            print("   ERROR: Evaluation failed")
            continue
            
        print(f"\n{model_name}")
        
        # 핵심 메트릭들만 출력
        metrics_to_show = [
            'obj_spatial_relation_oo_mv_accuracy',
            'obj_spatial_relation_oo_mv_exact_match', 
            'obj_spatial_relation_oo_mv_bleu',
            'obj_spatial_relation_oo_mv_rouge',
            'obj_spatial_relation_oo_mv_embedding_similarity',
            'obj_spatial_relation_oo_mv_spatial_similarity',
            'obj_spatial_relation_oo_mv_combined_similarity'
        ]
        
        # 메트릭명을 간단하게 변환
        metric_name_mapping = {
            'obj_spatial_relation_oo_mv_accuracy': 'avg_accuracy',
            'obj_spatial_relation_oo_mv_exact_match': 'avg_exact_match',
            'obj_spatial_relation_oo_mv_bleu': 'avg_bleu',
            'obj_spatial_relation_oo_mv_rouge': 'avg_rouge',
            'obj_spatial_relation_oo_mv_embedding_similarity': 'avg_embedding_similarity',
            'obj_spatial_relation_oo_mv_spatial_similarity': 'avg_spatial_similarity',
            'obj_spatial_relation_oo_mv_combined_similarity': 'avg_combined_similarity'
        }
        
        for metric_key in metrics_to_show:
            # breakpoint() 
            if metric_key in results:
                display_name = metric_name_mapping.get(metric_key, metric_key)
                value = results[metric_key]
                print(f"   {display_name}: {value:.4f}")
        
        # 전체 점수도 표시
        if 'overall' in results:
            print(f"   overall_score: {results['overall']:.4f}")

def main():
    """메인 실행 함수 (배치 평가용으로 수정)"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help-metrics":
        print_simplified_metric_documentation()
        return 0
    elif len(sys.argv) > 1 and sys.argv[1] == "--single":
        if len(sys.argv) > 2:
            jsonl_file = sys.argv[2]
        else:
            jsonl_file = "data_jsons/generated_predictions.jsonl"
            
        if not os.path.exists(jsonl_file):
            print(f"❌ Error: File not found: {jsonl_file}")
            return 1
            
        try:
            analyzer = run_complete_analysis(jsonl_file)
            print("\n✅ Single file analysis completed!")
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return 1
    else:
        base_dir = "Eval_Results"
        if len(sys.argv) > 1:
            base_dir = sys.argv[1]
            
        try:
            run_batch_evaluation(base_dir)
            print("\n✅ Batch evaluation completed!")
        except Exception as e:
            print(f"❌ Error during batch evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0









if __name__ == "__main__":
    exit(main())