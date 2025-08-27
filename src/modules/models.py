import torch
import pickle
from torch import Tensor
from tqdm.auto import tqdm
import torch.nn.functional as F 
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.sparse import diags
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import requests
import time

@dataclass
class OpusTranslationModelConfig:
    padding: bool
    model_name: str
    device: str
    dtype: str
    truncation: bool
    skip_special_tokens: bool


class OpusTranslationModel:

    def __init__(self, config: OpusTranslationModelConfig):
        self.config = config
        self.model = MarianMTModel.from_pretrained(
            self.config.model_name, 
            device_map=self.config.device, 
            torch_dtype=self.config.dtype
        )
        self.tokenizer = MarianTokenizer.from_pretrained(self.config.model_name)
        
    def translate(self, text: str) -> str:
        tokens = self.tokenizer(
            text, 
            padding=self.config.padding, 
            truncation=self.config.truncation, 
            return_tensors="pt"
        ).to(self.config.device)
        translated_tokens = self.model.generate(**tokens)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=self.config.skip_special_tokens)

        return translated_text


@dataclass
class SentenceEmbeddingConfig:
    device: str
    dtype: str
    model_id: str
    truncate_dim: Optional[int]
    convert_to_numpy: bool
    convert_to_tensor: bool
    use_prompt: bool = False
    prompt_config: Optional[Dict[str, str]] = None
    model_kwargs: Optional[Dict[str, Any]] = None


class SentenceEmbeddingModel:
    def __init__(self, config: SentenceEmbeddingConfig):
        super().__init__()
        self.config = config
        self.model_id = config.model_id
        self.device = config.device
        self.dtype = config.dtype
        self.truncate_dim = config.truncate_dim

        model_kwargs = config.model_kwargs or {}

        if "quantization_config" in model_kwargs:
            quant_config = model_kwargs["quantization_config"]
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant_config.get("load_in_4bit", True),
                bnb_4bit_compute_dtype=getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "float16")),
                bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True)
            )

        if "torch_dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            truncate_dim=self.truncate_dim,
            model_kwargs=model_kwargs
        )

    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        if self.config.use_prompt and prompt_name and self.config.prompt_config:
            if prompt_name in self.config.prompt_config:
                prompt_template = self.config.prompt_config[prompt_name]
                texts = [prompt_template.format(text=t) for t in texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=self.config.convert_to_numpy,
            convert_to_tensor=self.config.convert_to_tensor,
            show_progress_bar=True
        )
        return embeddings

    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        return self.model.similarity(query_embeddings, document_embeddings)

    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        query_embeddings = self.get_embeddings(queries, "classification")
        document_embeddings = self.get_embeddings(documents, "classification")
        return self.calculate_scores(query_embeddings, document_embeddings)
    

@dataclass
class LLMModelConfig:
    api_key: str
    model_name: str = "gemini-1.5-flash"
    api_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    temperature: float = 0.1
    max_tokens: int = 100
    top_p: float = 0.9


class LLMModel:
    
    def __init__(self, config: LLMModelConfig, prompt_path: str):
        self.config = config
        self.prompt_path = prompt_path
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _prepare_prompt(self, product_name: str, allowed_labels: List[str]) -> str:
        labels_text = "\n".join(allowed_labels)
        return self.prompt_template.replace("{{Product_Name}}", product_name).replace("{{LABEL_1}}\n{{LABEL_2}}\n{{LABEL_3}}\n{{LABEL_4}}\n{{LABEL_5}}\n....", labels_text)
    
    def _make_api_request(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
                "maxOutputTokens": self.config.max_tokens,
            }
        }
        
        url = f"{self.config.api_url}?key={self.config.api_key}"
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                return ""
        except Exception as e:
            print(f"API request failed: {e}")
            return ""
    
    def _parse_response(self, response: str, allowed_labels: List[str]) -> str:
        if not response:
            return allowed_labels[0] if allowed_labels else ""
        
        response = response.strip()
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for line in lines:
            line_clean = line.replace('```', '').strip()
            if line_clean in allowed_labels:
                return line_clean
        
        for label in allowed_labels:
            if label.lower() in response.lower():
                return label
        
        if lines:
            return lines[0].replace('```', '').strip()
        
        return allowed_labels[0] if allowed_labels else ""
    
    def predict(self, products_df: pd.DataFrame, allowed_labels: List[str], 
                product_text_col: str = "translated_name") -> pd.DataFrame:
        results = []
        
        for idx, row in products_df.iterrows():
            product_name = row[product_text_col]
            
            prompt = self._prepare_prompt(product_name, allowed_labels)
            response = self._make_api_request(prompt)
            predicted_label = self._parse_response(response, allowed_labels)
            
            result = {
                "product_id": row.get("id", idx),
                "product_text": product_name,
                "predicted_label_llm": predicted_label,
                "raw_response": response
            }
            results.append(result)

        time.sleep(0.5)
        
        return pd.DataFrame(results)

@dataclass
class ICFTDCBModelConfig:
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    k: int = 3
    class_name_col: str = "SegmentTitle"
    class_text_col: str = "SegmentDefinition"
    product_id_col: str = "id"
    product_text_col: str = "translated_name"


class ICFTDCBModel:
    
    def __init__(self, config: ICFTDCBModelConfig):
        self.config = config
        self.vectorizer = None
        self.class_centroids = None
        self.global_weights = None
        self.gpc_df = None
    
    def _prep_text(self, s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    
    def fit(self, gpc_df: pd.DataFrame):
        self.gpc_df = gpc_df.copy()
        self.gpc_df[self.config.class_name_col] = self._prep_text(self.gpc_df[self.config.class_name_col])
        
        self.vectorizer = CountVectorizer(
            ngram_range=self.config.ngram_range, 
            min_df=self.config.min_df
        )
        X_cls = self.vectorizer.fit_transform(self.gpc_df[self.config.class_name_col])
        C = X_cls.shape[0]
        
        cf = (X_cls > 0).sum(axis=0).A1 + 1e-9
        ICF = np.log(C / cf)
        
        col_sums = np.asarray(X_cls.sum(axis=0)).ravel() + 1e-9
        P = X_cls.multiply(1.0 / col_sums)
        TDCB = 1.0 - np.asarray(P.power(2).sum(axis=0)).ravel()
        
        self.global_weights = np.maximum(ICF * TDCB, 1e-9)
        
        self.class_centroids = normalize(X_cls @ diags(self.global_weights), norm="l2", axis=1)
    
    def predict(self, products_df: pd.DataFrame) -> pd.DataFrame:
        # if self.vectorizer is None or self.class_centroids is None:
        #     raise ValueError("Model must be fitted before prediction")
        
        products_df = products_df.copy()
        products_df[self.config.product_text_col] = self._prep_text(products_df[self.config.product_text_col])
        
        X_prod = self.vectorizer.transform(products_df[self.config.product_text_col])
        V_prod = normalize(X_prod @ diags(self.global_weights), norm="l2", axis=1)
        
        S = (V_prod @ self.class_centroids.T).toarray()
        topk_idx = (-S).argsort(1)[:, :self.config.k]
        topk_scores = np.take_along_axis(S, topk_idx, axis=1)
        
        rows = []
        for i in range(S.shape[0]):
            base = {}
            if self.config.product_id_col in products_df.columns:
                base["product_id"] = products_df.loc[i, self.config.product_id_col]
            base["product_text"] = products_df.loc[i, self.config.product_text_col]
            for j in range(self.config.k):
                base[f"predicted_label_icf"] = self.gpc_df.iloc[topk_idx[i,j]][self.config.class_name_col]
                base[f"score_{j+1}"] = float(topk_scores[i,j])
            rows.append(base)
        
        return pd.DataFrame(rows)


@dataclass
class TFIDFCentroidModelConfig:
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    max_df: float = 0.95
    max_features: Optional[int] = None
    k: int = 3
    class_name_col: str = "SegmentTitle"
    class_text_col: str = "SegmentDefinition"
    product_id_col: str = "id"
    product_text_col: str = "translated_name"

class TFIDFCentroidModel:
    
    def __init__(self, config: TFIDFCentroidModelConfig):
        self.config = config
        self.vectorizer = None
        self.class_centroids = None
        self.gpc_df = None
    
    def _prep_text(self, s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    
    def fit(self, gpc_df: pd.DataFrame):
        self.gpc_df = gpc_df.copy()
        self.gpc_df[self.config.class_name_col] = self._prep_text(self.gpc_df[self.config.class_name_col])
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            max_features=self.config.max_features
        )
        
        X_cls = self.vectorizer.fit_transform(self.gpc_df[self.config.class_name_col])
        
        self.class_centroids = normalize(X_cls, norm="l2", axis=1)
    
    def predict(self, products_df: pd.DataFrame) -> pd.DataFrame:
        if self.vectorizer is None or self.class_centroids is None:
            raise ValueError("Model must be fitted before prediction")
        
        products_df = products_df.copy()
        products_df[self.config.product_text_col] = self._prep_text(products_df[self.config.product_text_col])
        
        X_prod = self.vectorizer.transform(products_df[self.config.product_text_col])
        V_prod = normalize(X_prod, norm="l2", axis=1)
        
        similarities = cosine_similarity(V_prod, self.class_centroids)
        topk_idx = (-similarities).argsort(1)[:, :self.config.k]
        topk_scores = np.take_along_axis(similarities, topk_idx, axis=1)
        
        rows = []
        for i in range(similarities.shape[0]):
            base = {}
            if self.config.product_id_col in products_df.columns:
                base["product_id"] = products_df.loc[i, self.config.product_id_col]
            base["product_text"] = products_df.loc[i, self.config.product_text_col]
            base["predicted_label_idf"] = self.gpc_df.iloc[topk_idx[i,0]][self.config.class_name_col]
            for j in range(self.config.k):
                base[f"class_{j+1}_name"] = self.gpc_df.iloc[topk_idx[i,j]][self.config.class_name_col]
                base[f"score_{j+1}"] = float(topk_scores[i,j])
            rows.append(base)
        
        return pd.DataFrame(rows)

class EnsembleClassifier:
    
    def __init__(self, icf_config: ICFTDCBModelConfig,
                 tfidf_config: TFIDFCentroidModelConfig,
                 embedding_model: SentenceEmbeddingModel,
                 llm_model: LLMModel):
        self.icf_model = ICFTDCBModel(icf_config)
        self.tfidf_model = TFIDFCentroidModel(tfidf_config)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        self.gpc_df = None
        self.segment_embeddings = None
        self.segment_labels = None
    
    def fit(self, gpc_df: pd.DataFrame, segment_col: str = "SegmentTitle"):
        self.gpc_df = gpc_df.copy()
        
        unique_segments = gpc_df[segment_col].unique()
        self.segment_labels = unique_segments.tolist()
        
        segment_gpc = pd.DataFrame({
            'class_id': range(len(unique_segments)),
            'class_name': unique_segments,
            'class_text': unique_segments
        })
        
        self.icf_model.fit(segment_gpc)
        self.tfidf_model.fit(segment_gpc)
        
        self.segment_embeddings = self.embedding_model.get_embeddings(
            self.segment_labels, 
            prompt_name="classification"
        )
    
    def _cosine_similarity_predict(self, products_df: pd.DataFrame, 
                                   product_text_col: str = "translated_name") -> pd.DataFrame:
        product_texts = products_df[product_text_col].tolist()
        product_embeddings = self.embedding_model.get_embeddings(
            product_texts, 
            prompt_name="classification"
        )
        
        similarities = cosine_similarity(product_embeddings, self.segment_embeddings)
        
        results = []
        for i, row in products_df.iterrows():
            best_idx = np.argmax(similarities[i])
            predicted_label = self.segment_labels[best_idx]
            
            result = {
                "product_id": row.get("id", i),
                "product_text": row[product_text_col],
                "predicted_label_cosine": predicted_label,
                "confidence": float(similarities[i][best_idx])
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def predict_ensemble(self, products_df: pd.DataFrame, 
                        product_text_col: str = "translated_name",
                        voting_strategy: str = "majority") -> pd.DataFrame:
        
        icf_predictions = self.icf_model.predict(products_df)
        tfidf_predictions = self.tfidf_model.predict(products_df)
        cosine_predictions = self._cosine_similarity_predict(products_df, product_text_col)
        llm_predictions = self.llm_model.predict(products_df, self.segment_labels, product_text_col)
        
        ensemble_results = []
        
        for i in range(len(products_df)):
            product_id = products_df.iloc[i].get("id", i)
            product_text = products_df.iloc[i][product_text_col]
            
            predictions = {
                "icf": icf_predictions.iloc[i]["predicted_label_icf"],
                "tfidf": tfidf_predictions.iloc[i]["predicted_label_idf"],
                "cosine": cosine_predictions.iloc[i]["predicted_label_cosine"], 
                "llm": llm_predictions.iloc[i]["predicted_label_llm"]
            }
            
            if voting_strategy == "majority":
                votes = list(predictions.values())
                vote_counts = Counter(votes)
                final_prediction = vote_counts.most_common(1)[0][0]
                confidence = vote_counts.most_common(1)[0][1] / len(votes)
            
            elif voting_strategy == "weighted":
                icf_weight = 0.2
                tfidf_weight = 0.25
                cosine_weight = 0.35
                llm_weight = 0.2
                
                label_scores = {}
                weights_list = [icf_weight, tfidf_weight, cosine_weight, llm_weight]
                for pred, weight in zip(predictions.values(), weights_list):
                    if pred not in label_scores:
                        label_scores[pred] = 0
                    label_scores[pred] += weight
                
                final_prediction = max(label_scores.keys(), key=lambda x: label_scores[x])
                confidence = label_scores[final_prediction]
            
            else:
                final_prediction = predictions["llm"]
                confidence = 1.0
            
            result = {
                "product_id": product_id,
                "product_text": product_text,
                "final_prediction": final_prediction,
                "confidence": confidence,
                "icf_prediction": predictions["icf"],
                "tfidf_prediction": predictions["tfidf"],
                "cosine_prediction": predictions["cosine"],
                "llm_prediction": predictions["llm"]
            }
            
            ensemble_results.append(result)
        
        return pd.DataFrame(ensemble_results)


@dataclass
class DummyModelConfig:
    strategy: str


class DummyModel:

    def __init__(self, config: DummyModelConfig):
        self.model = DummyClassifier(strategy=config.strategy)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
            



