#!/usr/bin/env python3
"""
SCIO NLP Module

Erweiterte Natural Language Processing Faehigkeiten.
"""

from .advanced_nlp import (
    AdvancedNLP,
    TextProcessor,
    SimpleLemmatizer,
    SentimentAnalyzer,
    KeywordExtractor,
    SimpleNER,
    TextSummarizer,
    LanguageDetector,
    get_nlp,
    # Data Classes
    Token,
    Entity,
    Sentence,
    NLPAnalysis,
    # Enums
    EntityType,
    POSTag,
)

__all__ = [
    "AdvancedNLP",
    "TextProcessor",
    "SimpleLemmatizer",
    "SentimentAnalyzer",
    "KeywordExtractor",
    "SimpleNER",
    "TextSummarizer",
    "LanguageDetector",
    "get_nlp",
    "Token",
    "Entity",
    "Sentence",
    "NLPAnalysis",
    "EntityType",
    "POSTag",
]
