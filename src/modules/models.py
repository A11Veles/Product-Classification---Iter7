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
from sklearn.decomposition import PCA
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from constants import RANDOM_STATE, MODEL_PATH


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
class KMeansModelConfig:
    n_clusters: int
    topk: Optional[int] = None
    model_name: Optional[str] = None


class KMeansModels:
    
    def __init__(self, config: KMeansModelConfig) -> None:
        self.config = config
        self.topk = config.topk
        self.n_clusters = config.n_clusters
        self.model = (
            KMeans(self.n_clusters, random_state=RANDOM_STATE) 
            if config.model_name is None 
            else self.load_model()
        )

    def fit(self, X) -> None:
        self.model = self.model.fit(X)

    def cosine_similarity(self, cluster_embeddings: Tensor, classes_embeddings: Tensor) -> Tensor:
        cluster_embeddings = F.normalize(cluster_embeddings, p=2, dim=1)
        classes_embeddings = F.normalize(classes_embeddings, p=2, dim=1)

        scores = (cluster_embeddings @ classes_embeddings.T)

        return scores
    
    def get_topk_classes(self, scores: Tensor) -> Tensor:
        if self.topk is None:
            raise ValueError("You need to assing topk to use this function.")
         
        topk_classes = torch.topk(scores, k=self.topk)

        return topk_classes[1]
    
    def get_classes(self, scores: Tensor) -> Tensor:
        if self.topk is not None:
            classes = self.get_topk_classes(scores)
        else:
            classes = torch.argmax(scores, dim=1)

        return classes
    
    def get_centroid_classes(self, classes_embeddings: Tensor) -> Dict[int, Any]:
        centroid_classes = {}
        for i, centroid in tqdm(enumerate(self.model.cluster_centers_)):
            centroid = torch.tensor(centroid, dtype=classes_embeddings.dtype, device=classes_embeddings.device).unsqueeze(0)
            scores = self.cosine_similarity(centroid, classes_embeddings)
            classes = self.get_classes(scores)
            centroid_classes[i] = classes
        
        return centroid_classes
    
    def get_cluster_items(self, product_embeddings: Tensor) -> Dict[Tuple[int, int], Any]:
        labels = self.model.predict(product_embeddings.tolist())
        cluster_items = {i: [] for i in range(self.model.n_clusters)}
        for  i, (embedding, label) in enumerate(zip(product_embeddings, labels)):
            cluster_items[label].append((i, embedding))

        return cluster_items

    def get_cluster_classes(self, product_embeddings: Tensor, class_embeddings: Tensor) -> Dict[int, Any]:
        cluster_items = self.get_cluster_items(product_embeddings)

        clusters_class = {}
        for cluster, values in tqdm(cluster_items.items()):
            embeddings = [embedding for _, embedding in values]
            embeddings = torch.stack(embeddings)
            scores = self.cosine_similarity(embeddings, class_embeddings)
            classes = self.get_classes(scores)
            if classes.dim() > 1:
                classes = classes.flatten()
            counter = Counter(classes.tolist())
            mode, count = counter.most_common(1)[0]
            clusters_class[cluster] = mode, count, classes.shape[0], count/classes.shape[0]

        return clusters_class
    
    def save_model(self):
        with open(MODEL_PATH / self.config.model_name, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self) -> None:
        with open(MODEL_PATH / self.config.model_name, "rb") as f:
            model = pickle.load(f)

        return model


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
            



