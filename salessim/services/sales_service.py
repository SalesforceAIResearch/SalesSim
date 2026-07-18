#!/usr/bin/env python3

import hashlib
import os
import shutil
import sys
import json
import logging
from typing import List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from salessim.services.constants import Document

# Compatibility shim for loading old FAISS indices
import langchain_community.docstore
import langchain_core.documents
sys.modules['langchain.docstore'] = langchain_community.docstore
sys.modules['langchain.schema'] = langchain_core.documents
sys.modules['langchain.schema.document'] = langchain_core.documents

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str
    k: int = 4
    product_category: str | None = None

class DocumentResponse(BaseModel):
    page_content: str
    metadata: Dict

def _get_lookup_device() -> str:
    forced_device = os.environ.get("LOOKUP_DEVICE")
    if forced_device:
        return forced_device
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        return "cpu"
    return "cuda"


def _compute_data_hash(datapath: str) -> str:
    """SHA256 hash of all JSON files in datapath, for FAISS index invalidation."""
    h = hashlib.sha256()
    for fname in sorted(os.listdir(datapath)):
        if fname.endswith(".json"):
            with open(os.path.join(datapath, fname), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


class ProductLookupModule:
    def __init__(self, verbose=False):
        index_name = os.environ.get("PRODUCTS_INDEX_DIR", "products_faiss_index")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        device = _get_lookup_device()
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

        datapath = "data/products/"
        data_hash = _compute_data_hash(datapath)
        hash_file = os.path.join(index_name, "source_hash.txt")

        if os.path.isdir(index_name):
            stored_hash = ""
            if os.path.exists(hash_file):
                with open(hash_file) as fh:
                    stored_hash = fh.read().strip()
            if stored_hash == data_hash:
                logger.info(f"Loading local {index_name} (source hash matches)")
                self.db = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded product db")
                return
            else:
                logger.info("Product data changed (hash mismatch), rebuilding FAISS index...")
                shutil.rmtree(index_name)

        names = sorted(
            os.path.join(datapath, f)
            for f in os.listdir(datapath)
            if f.endswith(".json")
        )
        products = []
        for fpath in names:
            logger.info(f"Processing {fpath}")
            with open(fpath, 'r') as f:
                data = json.load(f)
                for k,values in data.items():
                    for v in values:
                        v['category'] = k
                        price_value = v.get('price')
                        price_raw = "" if price_value is None else str(price_value).strip()
                        if not price_raw:
                            v['price'] = "N/A"
                            v['price_float'] = None
                        else:
                            try:
                                v['price_float'] = float(price_raw.replace('$', '').replace(',', ''))
                            except ValueError:
                                logger.warning("Invalid price '%s' in %s; setting price_float=None", price_raw, fpath)
                                v['price_float'] = None
                        products.append(v)

        docs = []
        for i, product in enumerate(products):
            title = str(product.get('name', '')).strip()
            price = str(product.get('price', 'N/A')).strip()
            weight = str(product.get('weight', 'N/A')).strip()
            description = product['description']
            feature = ', '.join(product['features'])
            image = product.get('image') or product.get('image_url')
            image_line = f"Image: {image}" if image else ""
            product_doc = f"{title}\nPrice: {price}\nWeight: {weight}\n{description}\n{feature}"
            contents = f"Name: {title}\nPrice: {price}\nWeight: {weight}\nDescription: {description}\nFeatures: {feature}"
            if image_line:
                product_doc = f"{product_doc}\n{image_line}"
                contents = f"{contents}\n{image_line}"
            docs.append(Document(
                page_content=product_doc,
                metadata={
                    'title': title,
                    'id': str(i),
                    'contents': contents,
                    'category': product.get('category'),
                    **({'image': image} if image else {})
                },
                id=str(i)))
        logger.info(f"Processed {len(docs)} docs")

        self.db = FAISS.from_documents(docs, embeddings)
        self.db.save_local(index_name)
        with open(hash_file, "w") as fh:
            fh.write(data_hash)
        logger.info("Built and saved product FAISS index with source hash")

    def top_docs(self, query: str, k: int = 4, product_category: str = None):
        top_documents = self.db.similarity_search(query, k=k, filter={'category': product_category})
        return top_documents

class SearchBuyingGuide:
    def __init__(self, verbose=False):
        index_name = os.environ.get("GUIDES_INDEX_DIR", "guides_faiss_index//guides_faiss_index_all")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        device = _get_lookup_device()
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

        if os.path.isdir(index_name):
            logger.info("Loading local faiss index")
            self.db = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
        else:
            file_path = 'data/guides.json'
            guides = {}
            with open(file_path, 'r') as f:
                guides = json.load(f)
            logger.info(f"Loaded {len(guides)} buying guides")

            text_splitter = CharacterTextSplitter(separator="\n")
            docs = []
            for name, guide in guides.items():
                text = text_splitter.split_text(guide)
                docs.extend([Document(page_content=t, metadata={'title': name}) for t in text])
            logger.info(f"Processed {len(docs)} docs")

            self.db = FAISS.from_documents(docs, embeddings)
            self.db.save_local(index_name)
        logger.info("Loaded knowledge db")

    def top_docs(self, query: str, k: int = 4, product_category: str = None):
        # TODO: Maybe need to implement product category filtering.
        top_documents = self.db.similarity_search(query, k=k)
        return top_documents

# Service state
service_state = {
    "product_lookup_module": None,
    "buying_guide_module": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Lookup Service...")
    service_state["product_lookup_module"] = ProductLookupModule()
    service_state["buying_guide_module"] = SearchBuyingGuide()
    logger.info("Lookup Service started successfully")
    yield
    # Shutdown
    logger.info("Shutting down Lookup Service...")
    service_state["product_lookup_module"] = None
    service_state["buying_guide_module"] = None

app = FastAPI(title="Lookup Service", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "lookup_service"}

@app.post("/products/search", response_model=List[DocumentResponse])
async def search_products(request: SearchRequest):
    if service_state["product_lookup_module"] is None:
        raise HTTPException(status_code=503, detail="Product lookup service not initialized")

    try:
        docs = service_state["product_lookup_module"].top_docs(request.query, request.k, request.product_category)
        
        return [
            DocumentResponse(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc in docs
        ]
    except Exception as e:
        logger.error(f"Product search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/guides/search", response_model=List[DocumentResponse])
async def search_buying_guides(request: SearchRequest):
    if service_state["buying_guide_module"] is None:
        raise HTTPException(status_code=503, detail="Buying guide service not initialized")

    try:
        docs = service_state["buying_guide_module"].top_docs(request.query, request.k)
        return [
            DocumentResponse(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc in docs
        ]
    except Exception as e:
        logger.error(f"Buying guide search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003, log_level="info")
