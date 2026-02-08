#!/usr/bin/env python3
"""
SCIO - Knowledge Module
Wissensgraph, Ontologien und semantische Beziehungen
"""

from .knowledge_graph import KnowledgeGraph, Entity, Relation, get_knowledge_graph

__all__ = [
    'KnowledgeGraph',
    'Entity',
    'Relation',
    'get_knowledge_graph',
]
