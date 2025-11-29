#!/usr/bin/env python3

import os
import json
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from salessim.services.constants import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str
    k: int = 4

class DocumentResponse(BaseModel):
    page_content: str
    metadata: Dict

class RecommendedItemsRequest(BaseModel):
    candidates: List[Dict[str, Any]]  # List of product candidates with page_content and metadata
    response: str  # Sales agent response text
    sim_threshold: float = 0.70

class ProductLookupModule:
    def __init__(self, verbose=False):
        index_name = "products_faiss_index"
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Initialize embedder for similarity calculations
        self.embedder = SentenceTransformer(model_name)

        if os.path.isdir(index_name):
            logger.info(f"Loading local {index_name}")
            self.db = FAISS.load_local(index_name, embeddings)
        else:
            datapath = "data/products/"
            names = [datapath+d for d in os.listdir(datapath)]
            products = []
            for fpath in names:
                logger.info(f"Processing {fpath}")
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    for k,values in data.items():
                        for v in values:
                            v['category'] = k
                            v['price_float'] = float(v['price'].replace('$','').replace(',',''))
                            products.append(v)

            docs = []
            for i, product in enumerate(products):
                title = product['name'].strip()
                price = product['price'].strip()
                weight = product.get('weight', 'N/A').strip()
                description = product['description']
                feature = ', '.join(product['features'])
                product_doc = f"{title}\nPrice: {price}\nWeight: {weight}\n{description}\n{feature}"
                contents = f"Name: {title}\nPrice: {price}\nWeight: {weight}\nDescription: {description}\nFeatures: {feature}"
                docs.append(DocumentResponse(
                    page_content=product_doc,
                    metadata={'title': title, 'id': str(i), 'contents': contents}))
            logger.info(f"Processed {len(docs)} docs")

            self.db = FAISS.from_documents(docs, embeddings) 
            self.db.save_local(index_name)
        logger.info("Loaded product db")

    def _filter_similarity_candidates_to_sentences(self, candidates, sentence, sim_threshold):
        query_embedding = self.embedder.encode(sentence.lower(), convert_to_tensor=True)
        final_rec_items = []
        for item in candidates:
            candidate = item.metadata["title"].lower()
            if candidate in sentence:
                final_rec_items.append((1.0, item))
                continue
            candidate_embedding = self.embedder.encode(candidate, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, candidate_embedding)[0]
            sim_score = cos_scores.max().item()
            logging.info(f"{sim_score}: {candidate}")
            if sim_score > sim_threshold:
                logging.info(f"{candidate} was recommended.")
                final_rec_items.append((sim_score, item))
        recommended_items = [item for sim_score, item in sorted(final_rec_items)]
        return recommended_items

    def find_recommended_items_in_response(self, candidates, response, sim_threshold=0.70):
        sentences = sent_tokenize(response.lower())
        recommended_items = []
        recommended_titles = set()
        for sentence in sentences:
            mentioned_items = self._filter_similarity_candidates_to_sentences(candidates, sentence, sim_threshold)
            for item in mentioned_items:
                if item.metadata["title"] in recommended_titles:
                    continue
                else:
                    recommended_items.append(item)
                    recommended_titles.add(item.metadata["title"])
        return recommended_items
    
    def top_docs(self, query: str, k: int = 4):
        top_documents = self.db.similarity_search(query, k=k)
        return top_documents

class SearchBuyingGuide:
    def __init__(self, verbose=False):
        index_name = "guides_faiss_index"
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        if os.path.isdir(index_name):
            logger.info("Loading local faiss index")
            self.db = FAISS.load_local(index_name, embeddings)
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

    def top_docs(self, query: str, k: int = 4):
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
        docs = service_state["product_lookup_module"].top_docs(request.query, request.k)
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

@app.post("/sales/find_recommended_items", response_model=List[DocumentResponse])
async def find_recommended_items_endpoint(request: RecommendedItemsRequest):
    if service_state["product_lookup_module"] is None:
        raise HTTPException(status_code=503, detail="Product lookup service not initialized")

    try:
        # Convert dict candidates back to Document objects
        candidates = []
        for candidate_dict in request.candidates:
            doc = Document(
                page_content=candidate_dict["page_content"],
                metadata=candidate_dict["metadata"]
            )
            candidates.append(doc)

        # Find recommended items using ProductLookupModule
        recommended_items = service_state["product_lookup_module"].find_recommended_items_in_response(
            candidates,
            request.response,
            request.sim_threshold
        )

        return [
            DocumentResponse(
                page_content=item.page_content,
                metadata=item.metadata
            ) for item in recommended_items
        ]
    except Exception as e:
        logger.error(f"Find recommended items error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")