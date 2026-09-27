"""Retrieval module: reranking a shortlist of vector-search candidates.

Owns the cross-encoder wrapper (``cross_encoder.py``) used to reorder the
top-k matches from a search step by relevance before they reach a caller.
"""
