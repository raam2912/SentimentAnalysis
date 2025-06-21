import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pathlib import Path
import warnings


class SemanticNewsSearch:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_dir: str = "vector_indexes",
        index_type: str = "flat_l2"
    ):
        """
        Initialize the semantic search engine.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.index_dir = Path(index_dir)
        self.index_type = index_type
        self.config = {
            "model_name": model_name,
            "index_type": index_type,
            "created_at": None,
            "last_updated": None,
            "dimension": None
        }
        
        self.index_dir.mkdir(exist_ok=True)


    def build_index(
        self,
        headlines: List[str],
        metadata: List[Dict],
        ticker: str,
        **index_params
    ) -> None:
        """
        Create and save a FAISS index from headlines and metadata.
        """
        if len(headlines) != len(metadata):
            raise ValueError("Headlines and metadata must have the same length")

        self._validate_metadata(metadata)
        
        # Generate embeddings
        embeddings = self.model.encode(
            headlines,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create appropriate index type
        dim = embeddings.shape[1]
        self.config.update({
            "dimension": dim,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        })
        
        if self.index_type == "flat_l2":
            self.index = faiss.IndexFlatL2(dim)
        elif self.index_type == "ivf":
            nlist = index_params.get("nlist", min(100, len(headlines)))
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(embeddings)
            self.config["nlist"] = nlist
        elif self.index_type == "hnsw":
            M = index_params.get("M", 32)
            self.index = faiss.IndexHNSWFlat(dim, M)
            self.config["M"] = M
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        self.index.add(embeddings)
        self.metadata = metadata
        
        self._save_index(ticker)


    def _validate_metadata(self, metadata: List[Dict]) -> None:
        """Validate metadata structure and required fields"""
        if not metadata:
            return
            
        required_fields = {'headline', 'date'}
        first_fields = set(metadata[0].keys())
        
        if not required_fields.issubset(first_fields):
            missing = required_fields - first_fields
            raise ValueError(f"Metadata missing required fields: {missing}")

        for i, record in enumerate(metadata[1:]):
            if set(record.keys()) != first_fields:
                raise ValueError(
                    f"Metadata record {i+1} has different fields than first record"
                )


    def _save_index(self, ticker: str) -> None:
        """Save index, metadata and config to disk"""
        base_path = self.index_dir / ticker
        
        # Save FAISS index
        faiss.write_index(self.index, f"{base_path}_faiss.index")
        
        # Save metadata
        with open(f"{base_path}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
            
        # Save config
        with open(f"{base_path}_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)


    def load_index(self, ticker: str) -> None:
        """Load a FAISS index and its metadata from disk"""
        base_path = self.index_dir / ticker
        
        # Check files exist
        if not all(base_path.with_suffix(ext).exists() 
                  for ext in ["_faiss.index", "_metadata.json", "_config.json"]):
            raise FileNotFoundError(f"Index files not found for '{ticker}'")

        # Load index
        self.index = faiss.read_index(f"{base_path}_faiss.index")
        
        # Load metadata
        with open(f"{base_path}_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        # Load config
        with open(f"{base_path}_config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)


    def update_index(
        self,
        new_headlines: List[str],
        new_metadata: List[Dict],
        ticker: str
    ) -> None:
        """
        Update existing index with new data.
        """
        if not self.index:
            self.load_index(ticker)
            
        if len(new_headlines) != len(new_metadata):
            raise ValueError("New headlines and metadata must have the same length")
            
        # Validate new metadata matches existing structure
        if self.metadata:
            existing_fields = set(self.metadata[0].keys())
            new_fields = set(new_metadata[0].keys())
            if existing_fields != new_fields:
                warnings.warn(
                    f"New metadata fields don't match existing fields. "
                    f"Missing: {existing_fields - new_fields}. "
                    f"Extra: {new_fields - existing_fields}"
                )
        
        # Generate embeddings for new data
        new_embeddings = self.model.encode(
            new_headlines,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Verify embedding dimension matches index
        if new_embeddings.shape[1] != self.config["dimension"]:
            raise ValueError(
                f"New embeddings dimension {new_embeddings.shape[1]} "
                f"doesn't match index dimension {self.config['dimension']}"
            )
            
        self.index.add(new_embeddings)
        self.metadata.extend(new_metadata)
        self.config["last_updated"] = datetime.now().isoformat()
        
        self._save_index(ticker)


    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_func: Optional[callable] = None
    ) -> List[Dict]:
        """
        Search the index for similar items to the query.
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call `load_index()` first.")
            
        if not self.metadata:
            raise ValueError("No metadata loaded")
            
        # Apply metadata filter if provided
        search_metadata = self.metadata
        if filter_func is not None:
            search_metadata = [m for m in self.metadata if filter_func(m)]
            if not search_metadata:
                return []
                
        # For IVF indices, set number of clusters to search
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = min(10, self.index.nlist)
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"),
            min(top_k, len(search_metadata))
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(search_metadata):  # Check for valid index
                result = search_metadata[idx].copy()
                result["distance"] = float(dist)
                results.append(result)
                
        return sorted(results, key=lambda x: x["distance"])


    def get_index_stats(self) -> Dict:
        """Return statistics about the current index"""
        if not self.index:
            return {"status": "No index loaded"}
            
        stats = {
            "index_type": self.config.get("index_type"),
            "dimension": self.config.get("dimension"),
            "vectors": self.index.ntotal,
            "metadata_records": len(self.metadata),
            "created_at": self.config.get("created_at"),
            "last_updated": self.config.get("last_updated")
        }
        
        if isinstance(self.index, faiss.IndexIVFFlat):
            stats.update({
                "nlist": self.index.nlist,
                "nprobe": self.index.nprobe
            })
        elif isinstance(self.index, faiss.IndexHNSWFlat):
            stats["hnsw_M"] = self.index.hnsw.M
            
        return stats