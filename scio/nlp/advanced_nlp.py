#!/usr/bin/env python3
"""
SCIO - Advanced NLP Module

Erweiterte NLP-Faehigkeiten mit NLTK, spaCy und Transformers.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter


class EntityType(str, Enum):
    """Named Entity Typen"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    GPE = "GPE"  # Geo-Political Entity


class POSTag(str, Enum):
    """Part-of-Speech Tags"""
    NOUN = "NOUN"
    VERB = "VERB"
    ADJ = "ADJ"
    ADV = "ADV"
    PRON = "PRON"
    DET = "DET"
    ADP = "ADP"  # Adposition
    CONJ = "CONJ"
    PUNCT = "PUNCT"
    NUM = "NUM"
    PROPN = "PROPN"  # Proper Noun
    INTJ = "INTJ"  # Interjection


@dataclass
class Token:
    """Repraesentiert ein Token"""
    text: str
    lemma: str = ""
    pos: str = ""
    tag: str = ""
    dep: str = ""  # Dependency
    is_stop: bool = False
    is_punct: bool = False
    is_digit: bool = False
    is_alpha: bool = True


@dataclass
class Entity:
    """Named Entity"""
    text: str
    label: str
    start: int = 0
    end: int = 0
    confidence: float = 1.0


@dataclass
class Sentence:
    """Ein Satz"""
    text: str
    tokens: List[Token] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    sentiment_score: float = 0.0


@dataclass
class NLPAnalysis:
    """Vollstaendige NLP-Analyse"""
    text: str
    sentences: List[Sentence] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    summary: str = ""
    language: str = "en"
    sentiment: float = 0.0
    topics: List[str] = field(default_factory=list)


class TextProcessor:
    """Basis-Textverarbeitung ohne externe Bibliotheken"""

    # Deutsche und englische Stoppwoerter
    STOPWORDS_EN = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "i", "you", "he", "she", "it",
        "we", "they", "what", "which", "who", "whom", "whose", "where",
        "when", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "also", "now"
    }

    STOPWORDS_DE = {
        "der", "die", "das", "ein", "eine", "und", "oder", "aber", "in",
        "auf", "an", "zu", "fuer", "von", "mit", "bei", "nach", "als",
        "ist", "war", "sind", "waren", "sein", "haben", "hat", "hatte",
        "werden", "wird", "wurde", "kann", "koennen", "muss", "muessen",
        "soll", "sollen", "will", "wollen", "darf", "duerfen", "dieser",
        "diese", "dieses", "jener", "jene", "jenes", "ich", "du", "er",
        "sie", "es", "wir", "ihr", "was", "wer", "wo", "wann", "warum",
        "wie", "alle", "jeder", "jede", "jedes", "einige", "andere",
        "mehr", "nur", "auch", "schon", "noch", "sehr", "hier", "dort"
    }

    def __init__(self, language: str = "en"):
        self.language = language
        self.stopwords = self.STOPWORDS_EN if language == "en" else self.STOPWORDS_DE

    def tokenize(self, text: str) -> List[str]:
        """Einfache Tokenisierung"""
        # Punktuation separieren
        text = re.sub(r'([.,!?;:"\'])', r' \1 ', text)
        tokens = text.split()
        return [t.strip() for t in tokens if t.strip()]

    def sentence_tokenize(self, text: str) -> List[str]:
        """Satz-Tokenisierung"""
        # Einfache Regel-basierte Satz-Erkennung
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Entfernt Stoppwoerter"""
        return [t for t in tokens if t.lower() not in self.stopwords]

    def normalize(self, text: str) -> str:
        """Normalisiert Text"""
        # Kleinschreibung
        text = text.lower()
        # Mehrfache Leerzeichen entfernen
        text = re.sub(r'\s+', ' ', text)
        # URLs entfernen
        text = re.sub(r'https?://\S+', '', text)
        # Email entfernen
        text = re.sub(r'\S+@\S+', '', text)
        return text.strip()

    def extract_ngrams(self, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        """Extrahiert N-Gramme"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


class SimpleLemmatizer:
    """Einfacher regelbasierter Lemmatizer"""

    # Englische Suffix-Regeln
    SUFFIX_RULES_EN = [
        ("ies", "y"),
        ("ied", "y"),
        ("es", ""),
        ("ed", ""),
        ("ing", ""),
        ("s", ""),
        ("ly", ""),
        ("ment", ""),
        ("ness", ""),
        ("tion", ""),
        ("able", ""),
        ("ible", ""),
    ]

    # Deutsche Suffix-Regeln
    SUFFIX_RULES_DE = [
        ("ung", ""),
        ("heit", ""),
        ("keit", ""),
        ("lich", ""),
        ("isch", ""),
        ("ig", ""),
        ("en", ""),
        ("er", ""),
        ("es", ""),
        ("e", ""),
    ]

    def __init__(self, language: str = "en"):
        self.language = language
        self.rules = self.SUFFIX_RULES_EN if language == "en" else self.SUFFIX_RULES_DE

    def lemmatize(self, word: str) -> str:
        """Lemmatisiert ein Wort"""
        word = word.lower()

        for suffix, replacement in self.rules:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)] + replacement

        return word


class SentimentAnalyzer:
    """Regelbasierte Sentiment-Analyse"""

    # Sentiment-Lexikon (vereinfacht)
    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "awesome", "love", "like", "best", "happy", "joy", "beautiful",
        "perfect", "success", "win", "positive", "brilliant", "outstanding",
        "superb", "terrific", "fabulous", "marvelous", "exceptional",
        "gut", "super", "toll", "wunderbar", "fantastisch", "perfekt",
        "gluecklich", "freude", "liebe", "erfolg", "gewinn", "positiv"
    }

    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "poor", "worst", "hate",
        "dislike", "sad", "angry", "fail", "failure", "negative", "wrong",
        "ugly", "stupid", "boring", "disappointing", "frustrating",
        "schlecht", "schrecklich", "furchtbar", "traurig", "wuetend",
        "versagen", "negativ", "falsch", "haesslich", "dumm", "langweilig"
    }

    INTENSIFIERS = {"very", "really", "extremely", "absolutely", "totally", "completely", "sehr", "wirklich", "extrem"}
    NEGATIONS = {"not", "no", "never", "neither", "nobody", "nothing", "nicht", "kein", "nie", "niemals"}

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analysiert Sentiment eines Textes"""
        words = text.lower().split()

        positive_count = 0
        negative_count = 0
        intensifier_active = False
        negation_active = False

        for i, word in enumerate(words):
            # Intensifier pruefen
            if word in self.INTENSIFIERS:
                intensifier_active = True
                continue

            # Negation pruefen
            if word in self.NEGATIONS:
                negation_active = True
                continue

            # Sentiment-Wort pruefen
            multiplier = 1.5 if intensifier_active else 1.0

            if word in self.POSITIVE_WORDS:
                if negation_active:
                    negative_count += multiplier
                else:
                    positive_count += multiplier
            elif word in self.NEGATIVE_WORDS:
                if negation_active:
                    positive_count += multiplier
                else:
                    negative_count += multiplier

            # Reset nach 2 Woertern
            if i > 0 and words[i-1] not in self.INTENSIFIERS:
                intensifier_active = False
            if i > 1 and words[i-2] not in self.NEGATIONS:
                negation_active = False

        total = positive_count + negative_count
        if total == 0:
            score = 0.0
            label = "neutral"
        else:
            score = (positive_count - negative_count) / total
            if score > 0.1:
                label = "positive"
            elif score < -0.1:
                label = "negative"
            else:
                label = "neutral"

        return {
            "score": score,
            "label": label,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "confidence": abs(score)
        }


class KeywordExtractor:
    """Keyword-Extraktion mit TF-IDF-aehnlicher Methode"""

    def __init__(self, language: str = "en"):
        self.processor = TextProcessor(language)
        self.lemmatizer = SimpleLemmatizer(language)

    def extract(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extrahiert Keywords aus Text"""
        # Tokenisieren
        tokens = self.processor.tokenize(text)

        # Stoppwoerter entfernen und lemmatisieren
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens
                  if t.lower() not in self.processor.stopwords
                  and len(t) > 2
                  and t.isalpha()]

        # Haeufigkeit zaehlen
        freq = Counter(tokens)

        # Normalisieren
        if not freq:
            return []

        max_freq = max(freq.values())
        keywords = [(word, count / max_freq) for word, count in freq.most_common(top_n)]

        return keywords

    def extract_phrases(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Extrahiert Schluesselphrasen (Bigrams)"""
        tokens = self.processor.tokenize(text)
        tokens = [t.lower() for t in tokens if t.isalpha() and len(t) > 2]

        # Bigrams erstellen
        bigrams = self.processor.extract_ngrams(tokens, 2)

        # Filtern (keine Stoppwoerter)
        bigrams = [bg for bg in bigrams
                   if bg[0] not in self.processor.stopwords
                   and bg[1] not in self.processor.stopwords]

        # Zaehlen
        freq = Counter(bigrams)

        if not freq:
            return []

        max_freq = max(freq.values())
        phrases = [(" ".join(bg), count / max_freq) for bg, count in freq.most_common(top_n)]

        return phrases


class SimpleNER:
    """Einfache Named Entity Recognition (regelbasiert)"""

    # Bekannte Entitaeten
    KNOWN_PERSONS = {"elon musk", "jeff bezos", "bill gates", "mark zuckerberg", "tim cook"}
    KNOWN_ORGS = {"google", "apple", "microsoft", "amazon", "facebook", "meta", "tesla", "nvidia", "openai"}
    KNOWN_LOCATIONS = {"new york", "los angeles", "san francisco", "london", "berlin", "paris", "tokyo"}

    def __init__(self):
        # Patterns
        self.patterns = {
            EntityType.DATE: r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b',
            EntityType.TIME: r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            EntityType.MONEY: r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|CHF)\b',
            EntityType.PERCENT: r'\b\d+(?:\.\d+)?%\b',
        }

    def extract(self, text: str) -> List[Entity]:
        """Extrahiert Named Entities"""
        entities = []
        text_lower = text.lower()

        # Pattern-basierte Extraktion
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    label=entity_type.value,
                    start=match.start(),
                    end=match.end()
                ))

        # Bekannte Entitaeten
        for person in self.KNOWN_PERSONS:
            if person in text_lower:
                idx = text_lower.find(person)
                entities.append(Entity(
                    text=text[idx:idx+len(person)],
                    label=EntityType.PERSON.value,
                    start=idx,
                    end=idx+len(person)
                ))

        for org in self.KNOWN_ORGS:
            if org in text_lower:
                idx = text_lower.find(org)
                entities.append(Entity(
                    text=text[idx:idx+len(org)],
                    label=EntityType.ORGANIZATION.value,
                    start=idx,
                    end=idx+len(org)
                ))

        for loc in self.KNOWN_LOCATIONS:
            if loc in text_lower:
                idx = text_lower.find(loc)
                entities.append(Entity(
                    text=text[idx:idx+len(loc)],
                    label=EntityType.LOCATION.value,
                    start=idx,
                    end=idx+len(loc)
                ))

        # Capitalized Words als potentielle Entities (heuristisch)
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            word = match.group()
            if word.lower() not in self.KNOWN_PERSONS and word.lower() not in self.KNOWN_ORGS:
                # Koennte Person oder Org sein
                entities.append(Entity(
                    text=word,
                    label="UNKNOWN",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.5
                ))

        return entities


class TextSummarizer:
    """Extraktive Textzusammenfassung"""

    def __init__(self, language: str = "en"):
        self.processor = TextProcessor(language)
        self.keyword_extractor = KeywordExtractor(language)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Erstellt eine Zusammenfassung"""
        sentences = self.processor.sentence_tokenize(text)

        if len(sentences) <= num_sentences:
            return text

        # Keywords extrahieren
        keywords = set(kw for kw, _ in self.keyword_extractor.extract(text, top_n=20))

        # Saetze nach Wichtigkeit bewerten
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            tokens = self.processor.tokenize(sentence.lower())

            # Score basierend auf Keywords
            keyword_score = sum(1 for t in tokens if t in keywords) / max(len(tokens), 1)

            # Position-Score (erste und letzte Saetze wichtiger)
            if i == 0:
                position_score = 1.0
            elif i == len(sentences) - 1:
                position_score = 0.8
            else:
                position_score = 0.5

            # Laengen-Score (mittlere Laenge bevorzugt)
            length_score = min(len(sentence) / 100, 1.0) * min(100 / max(len(sentence), 1), 1.0)

            total_score = keyword_score * 0.5 + position_score * 0.3 + length_score * 0.2
            scored_sentences.append((sentence, total_score, i))

        # Top-Saetze auswaehlen (in Original-Reihenfolge)
        top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: x[2])

        return " ".join(s[0] for s in top_sentences)


class LanguageDetector:
    """Einfache Spracherkennung"""

    # Charakteristische Woerter
    LANGUAGE_MARKERS = {
        "en": {"the", "is", "and", "of", "to", "in", "for", "with", "that", "this"},
        "de": {"der", "die", "das", "und", "ist", "ein", "eine", "fuer", "mit", "von"},
        "fr": {"le", "la", "les", "de", "et", "est", "un", "une", "pour", "avec"},
        "es": {"el", "la", "los", "las", "de", "y", "es", "un", "una", "para"},
        "it": {"il", "la", "di", "e", "che", "un", "una", "per", "con", "sono"},
    }

    def detect(self, text: str) -> str:
        """Erkennt die Sprache"""
        words = set(text.lower().split())

        scores = {}
        for lang, markers in self.LANGUAGE_MARKERS.items():
            score = len(words & markers)
            scores[lang] = score

        if not scores or max(scores.values()) == 0:
            return "unknown"

        return max(scores, key=scores.get)


class AdvancedNLP:
    """Hauptklasse fuer erweiterte NLP-Funktionen"""

    def __init__(self, language: str = "auto"):
        self._language = language
        self._detector = LanguageDetector()

        # Lazy-loaded Components
        self._processor: Optional[TextProcessor] = None
        self._lemmatizer: Optional[SimpleLemmatizer] = None
        self._sentiment: Optional[SentimentAnalyzer] = None
        self._keywords: Optional[KeywordExtractor] = None
        self._ner: Optional[SimpleNER] = None
        self._summarizer: Optional[TextSummarizer] = None

        # Optional: spaCy und NLTK
        self._spacy_nlp = None
        self._nltk_loaded = False

    def _get_language(self, text: str) -> str:
        """Ermittelt Sprache"""
        if self._language == "auto":
            return self._detector.detect(text)
        return self._language

    def _init_components(self, language: str):
        """Initialisiert Komponenten fuer Sprache"""
        if self._processor is None or self._processor.language != language:
            self._processor = TextProcessor(language)
            self._lemmatizer = SimpleLemmatizer(language)
            self._sentiment = SentimentAnalyzer()
            self._keywords = KeywordExtractor(language)
            self._ner = SimpleNER()
            self._summarizer = TextSummarizer(language)

    def _try_load_spacy(self, language: str) -> bool:
        """Versucht spaCy zu laden"""
        if self._spacy_nlp is not None:
            return True

        try:
            import spacy
            model = "en_core_web_sm" if language == "en" else f"{language}_core_news_sm"
            self._spacy_nlp = spacy.load(model)
            return True
        except (ImportError, OSError):
            return False

    def _try_load_nltk(self) -> bool:
        """Versucht NLTK zu laden"""
        if self._nltk_loaded:
            return True

        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
            self._nltk_loaded = True
            return True
        except (ImportError, LookupError):
            return False

    def analyze(self, text: str) -> NLPAnalysis:
        """Vollstaendige NLP-Analyse"""
        language = self._get_language(text)
        self._init_components(language)

        # Versuche spaCy fuer bessere Ergebnisse
        if self._try_load_spacy(language):
            return self._analyze_with_spacy(text, language)

        # Fallback auf regelbasierte Analyse
        return self._analyze_rule_based(text, language)

    def _analyze_rule_based(self, text: str, language: str) -> NLPAnalysis:
        """Regelbasierte Analyse"""
        # Saetze
        sentences_text = self._processor.sentence_tokenize(text)
        sentences = []

        for sent_text in sentences_text:
            tokens = self._processor.tokenize(sent_text)
            token_objs = [
                Token(
                    text=t,
                    lemma=self._lemmatizer.lemmatize(t),
                    is_stop=t.lower() in self._processor.stopwords,
                    is_punct=not t.isalnum(),
                    is_digit=t.isdigit(),
                    is_alpha=t.isalpha()
                )
                for t in tokens
            ]

            sent_sentiment = self._sentiment.analyze(sent_text)

            sentences.append(Sentence(
                text=sent_text,
                tokens=token_objs,
                sentiment_score=sent_sentiment["score"]
            ))

        # Gesamtanalyse
        entities = self._ner.extract(text)
        keywords = self._keywords.extract(text)
        summary = self._summarizer.summarize(text)
        overall_sentiment = self._sentiment.analyze(text)

        return NLPAnalysis(
            text=text,
            sentences=sentences,
            entities=entities,
            keywords=keywords,
            summary=summary,
            language=language,
            sentiment=overall_sentiment["score"]
        )

    def _analyze_with_spacy(self, text: str, language: str) -> NLPAnalysis:
        """Analyse mit spaCy"""
        doc = self._spacy_nlp(text)

        sentences = []
        for sent in doc.sents:
            tokens = [
                Token(
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    tag=token.tag_,
                    dep=token.dep_,
                    is_stop=token.is_stop,
                    is_punct=token.is_punct,
                    is_digit=token.is_digit,
                    is_alpha=token.is_alpha
                )
                for token in sent
            ]

            sent_sentiment = self._sentiment.analyze(sent.text)

            sentences.append(Sentence(
                text=sent.text,
                tokens=tokens,
                sentiment_score=sent_sentiment["score"]
            ))

        # spaCy Entities
        entities = [
            Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            )
            for ent in doc.ents
        ]

        keywords = self._keywords.extract(text)
        summary = self._summarizer.summarize(text)
        overall_sentiment = self._sentiment.analyze(text)

        return NLPAnalysis(
            text=text,
            sentences=sentences,
            entities=entities,
            keywords=keywords,
            summary=summary,
            language=language,
            sentiment=overall_sentiment["score"]
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenisiert Text"""
        self._init_components(self._get_language(text))
        return self._processor.tokenize(text)

    def extract_entities(self, text: str) -> List[Entity]:
        """Extrahiert Named Entities"""
        language = self._get_language(text)

        if self._try_load_spacy(language):
            doc = self._spacy_nlp(text)
            return [
                Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
                for ent in doc.ents
            ]

        self._init_components(language)
        return self._ner.extract(text)

    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Analysiert Sentiment"""
        self._init_components(self._get_language(text))
        return self._sentiment.analyze(text)

    def get_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extrahiert Keywords"""
        self._init_components(self._get_language(text))
        return self._keywords.extract(text, top_n)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Fasst Text zusammen"""
        self._init_components(self._get_language(text))
        return self._summarizer.summarize(text, num_sentences)

    def detect_language(self, text: str) -> str:
        """Erkennt Sprache"""
        return self._detector.detect(text)


# Singleton
_nlp: Optional[AdvancedNLP] = None


def get_nlp() -> AdvancedNLP:
    """Gibt NLP-Singleton zurueck"""
    global _nlp
    if _nlp is None:
        _nlp = AdvancedNLP()
    return _nlp
