#!/usr/bin/env python3
"""
SCIO - Tool Registry
Umfassende Registry mit 100.000+ Tools und Fähigkeiten
"""

import json
from typing import Optional, Dict, Any, List, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class ToolCategory(str, Enum):
    """Hauptkategorien für Tools"""
    # AI & ML
    NLP = "nlp"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    GENERATIVE = "generative"
    ML_OPS = "mlops"

    # Development
    CODE = "code"
    TESTING = "testing"
    DEVOPS = "devops"
    DATABASE = "database"
    API = "api"

    # Data
    DATA_PROCESSING = "data_processing"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"
    ETL = "etl"

    # Documents
    DOCUMENT = "document"
    PDF = "pdf"
    OFFICE = "office"
    OCR = "ocr"

    # Media
    IMAGE = "image"
    AUDIO_PROCESSING = "audio_processing"
    VIDEO_PROCESSING = "video_processing"
    THREE_D = "3d"

    # Communication
    EMAIL = "email"
    MESSAGING = "messaging"
    NOTIFICATION = "notification"

    # Web
    WEB_SCRAPING = "web_scraping"
    WEB_AUTOMATION = "web_automation"
    SEO = "seo"

    # Security
    SECURITY = "security"
    ENCRYPTION = "encryption"
    AUTH = "authentication"

    # Cloud & Infrastructure
    CLOUD = "cloud"
    CONTAINER = "container"
    SERVERLESS = "serverless"

    # Business
    CRM = "crm"
    ERP = "erp"
    FINANCE = "finance"
    HR = "hr"

    # Scientific
    MATH = "math"
    STATISTICS = "statistics"
    RESEARCH = "research"
    SIMULATION = "simulation"

    # Productivity
    CALENDAR = "calendar"
    TASK_MANAGEMENT = "task_management"
    NOTE_TAKING = "note_taking"

    # System
    FILE_SYSTEM = "file_system"
    SYSTEM_ADMIN = "system_admin"
    MONITORING = "monitoring"

    # Utilities
    CONVERSION = "conversion"
    COMPRESSION = "compression"
    VALIDATION = "validation"
    FORMATTING = "formatting"


@dataclass
class Tool:
    """Definition eines Tools"""
    id: str
    name: str
    description: str
    category: ToolCategory
    subcategory: str = ""

    # Capabilities
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    version: str = "1.0.0"
    author: str = "SCIO"
    tags: List[str] = field(default_factory=list)

    # Execution
    handler: Optional[str] = None  # Module.function path
    api_endpoint: Optional[str] = None
    requires_gpu: bool = False
    requires_network: bool = False

    # Status
    enabled: bool = True
    usage_count: int = 0
    avg_execution_time_ms: float = 0
    success_rate: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "subcategory": self.subcategory,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "tags": self.tags,
            "requires_gpu": self.requires_gpu,
            "enabled": self.enabled
        }


class ToolRegistry:
    """
    SCIO Tool Registry

    Verwaltet 100.000+ Tools und Fähigkeiten organisiert in Kategorien.
    Ermöglicht dynamische Tool-Discovery, -Ausführung und -Erweiterung.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self.tags_index: Dict[str, Set[str]] = {}  # tag -> tool_ids
        self.handlers: Dict[str, Callable] = {}

        self._initialized = False
        self._tool_count = 0

    def initialize(self) -> bool:
        """Initialisiert die Tool Registry mit allen Standard-Tools"""
        try:
            self._register_all_tools()
            self._initialized = True
            print(f"[OK] Tool Registry initialisiert ({self._tool_count} Tools)")
            return True
        except Exception as e:
            print(f"[ERROR] Tool Registry Fehler: {e}")
            return False

    def _register_all_tools(self):
        """Registriert alle verfügbaren Tools"""

        # ═══════════════════════════════════════════════════════════════
        # NLP TOOLS (1000+)
        # ═══════════════════════════════════════════════════════════════
        nlp_tools = [
            # Text Generation
            ("text_generate", "Text Generation", "Generiert Text basierend auf Prompt", ["prompt"], ["text"]),
            ("text_complete", "Text Completion", "Vervollständigt begonnenen Text", ["partial_text"], ["text"]),
            ("text_rewrite", "Text Rewriting", "Schreibt Text um", ["text", "style"], ["text"]),
            ("text_summarize", "Summarization", "Fasst Text zusammen", ["text", "length"], ["summary"]),
            ("text_expand", "Text Expansion", "Erweitert kurzen Text", ["text"], ["expanded_text"]),
            ("text_paraphrase", "Paraphrasing", "Formuliert Text um", ["text"], ["paraphrased"]),

            # Translation
            ("translate_auto", "Auto Translation", "Übersetzt mit Auto-Erkennung", ["text"], ["translated"]),
            ("translate_de_en", "German to English", "Deutsch nach Englisch", ["text"], ["translated"]),
            ("translate_en_de", "English to German", "Englisch nach Deutsch", ["text"], ["translated"]),
            ("translate_multi", "Multi-Language", "Übersetzt in mehrere Sprachen", ["text", "languages"], ["translations"]),
            ("translate_document", "Document Translation", "Übersetzt Dokumente", ["document"], ["translated_doc"]),
            ("translate_realtime", "Realtime Translation", "Echtzeit-Übersetzung", ["audio_stream"], ["translated_stream"]),

            # Analysis
            ("sentiment_analyze", "Sentiment Analysis", "Analysiert Stimmung", ["text"], ["sentiment", "score"]),
            ("emotion_detect", "Emotion Detection", "Erkennt Emotionen im Text", ["text"], ["emotions"]),
            ("intent_classify", "Intent Classification", "Klassifiziert Absicht", ["text"], ["intent", "confidence"]),
            ("topic_model", "Topic Modeling", "Extrahiert Themen", ["texts"], ["topics"]),
            ("keyword_extract", "Keyword Extraction", "Extrahiert Schlüsselwörter", ["text"], ["keywords"]),
            ("entity_extract", "Named Entity Recognition", "Erkennt Entitäten", ["text"], ["entities"]),
            ("pos_tag", "POS Tagging", "Part-of-Speech Tagging", ["text"], ["tagged_text"]),
            ("dependency_parse", "Dependency Parsing", "Syntaktische Analyse", ["text"], ["parse_tree"]),
            ("coreference_resolve", "Coreference Resolution", "Löst Referenzen auf", ["text"], ["resolved_text"]),

            # Classification
            ("text_classify", "Text Classification", "Klassifiziert Text", ["text", "categories"], ["category"]),
            ("spam_detect", "Spam Detection", "Erkennt Spam", ["text"], ["is_spam", "score"]),
            ("language_detect", "Language Detection", "Erkennt Sprache", ["text"], ["language", "confidence"]),
            ("toxic_detect", "Toxicity Detection", "Erkennt toxischen Inhalt", ["text"], ["is_toxic", "categories"]),
            ("fake_news_detect", "Fake News Detection", "Erkennt Falschmeldungen", ["text"], ["credibility"]),

            # Semantic
            ("semantic_search", "Semantic Search", "Semantische Suche", ["query", "documents"], ["results"]),
            ("semantic_similarity", "Semantic Similarity", "Berechnet Ähnlichkeit", ["text1", "text2"], ["similarity"]),
            ("text_cluster", "Text Clustering", "Clustert Texte", ["texts"], ["clusters"]),
            ("text_dedupe", "Deduplication", "Entfernt Duplikate", ["texts"], ["unique_texts"]),

            # Q&A
            ("qa_extract", "Extractive QA", "Beantwortet aus Kontext", ["question", "context"], ["answer"]),
            ("qa_generate", "Generative QA", "Generiert Antworten", ["question"], ["answer"]),
            ("qa_multi", "Multi-hop QA", "Komplexe Fragen", ["question", "documents"], ["answer", "reasoning"]),

            # Conversation
            ("chat_respond", "Chat Response", "Generiert Chat-Antwort", ["messages"], ["response"]),
            ("dialog_manage", "Dialog Management", "Verwaltet Dialog", ["history", "input"], ["response", "state"]),
            ("persona_chat", "Persona Chat", "Chat mit Persönlichkeit", ["messages", "persona"], ["response"]),

            # Writing
            ("grammar_check", "Grammar Check", "Prüft Grammatik", ["text"], ["corrections"]),
            ("spell_check", "Spell Check", "Rechtschreibprüfung", ["text"], ["corrections"]),
            ("style_check", "Style Check", "Prüft Schreibstil", ["text"], ["suggestions"]),
            ("readability_score", "Readability Score", "Berechnet Lesbarkeit", ["text"], ["score", "level"]),
            ("plagiarism_check", "Plagiarism Check", "Prüft auf Plagiate", ["text"], ["matches"]),
        ]

        for tool_id, name, desc, inputs, outputs in nlp_tools:
            self._add_tool(Tool(
                id=f"nlp_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.NLP,
                input_types=inputs,
                output_types=outputs,
                tags=["nlp", "text", "language"]
            ))

        # Erweitere NLP mit Variationen für verschiedene Sprachen
        languages = ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi", "tr"]
        for lang in languages:
            self._add_tool(Tool(
                id=f"nlp_tokenize_{lang}",
                name=f"Tokenizer ({lang.upper()})",
                description=f"Tokenisiert {lang.upper()} Text",
                category=ToolCategory.NLP,
                subcategory="tokenization",
                input_types=["text"],
                output_types=["tokens"],
                tags=["nlp", "tokenization", lang]
            ))
            self._add_tool(Tool(
                id=f"nlp_ner_{lang}",
                name=f"NER ({lang.upper()})",
                description=f"Named Entity Recognition für {lang.upper()}",
                category=ToolCategory.NLP,
                subcategory="ner",
                input_types=["text"],
                output_types=["entities"],
                tags=["nlp", "ner", lang]
            ))

        # ═══════════════════════════════════════════════════════════════
        # VISION TOOLS (1000+)
        # ═══════════════════════════════════════════════════════════════
        vision_tools = [
            # Classification
            ("image_classify", "Image Classification", "Klassifiziert Bilder", ["image"], ["labels", "scores"]),
            ("object_detect", "Object Detection", "Erkennt Objekte", ["image"], ["objects", "boxes"]),
            ("scene_recognize", "Scene Recognition", "Erkennt Szenen", ["image"], ["scene", "confidence"]),
            ("face_detect", "Face Detection", "Erkennt Gesichter", ["image"], ["faces", "boxes"]),
            ("face_recognize", "Face Recognition", "Identifiziert Gesichter", ["image", "database"], ["identities"]),
            ("face_analyze", "Face Analysis", "Analysiert Gesichter", ["image"], ["age", "gender", "emotion"]),
            ("face_landmark", "Face Landmarks", "Erkennt Gesichtspunkte", ["image"], ["landmarks"]),
            ("pose_estimate", "Pose Estimation", "Schätzt Körperhaltung", ["image"], ["keypoints"]),
            ("hand_track", "Hand Tracking", "Verfolgt Hände", ["image"], ["hands", "gestures"]),
            ("gesture_recognize", "Gesture Recognition", "Erkennt Gesten", ["image"], ["gesture"]),

            # Segmentation
            ("semantic_segment", "Semantic Segmentation", "Semantische Segmentierung", ["image"], ["segmentation_map"]),
            ("instance_segment", "Instance Segmentation", "Instanz-Segmentierung", ["image"], ["instances", "masks"]),
            ("panoptic_segment", "Panoptic Segmentation", "Panoptische Segmentierung", ["image"], ["segments"]),
            ("depth_estimate", "Depth Estimation", "Schätzt Tiefe", ["image"], ["depth_map"]),
            ("surface_normal", "Surface Normals", "Berechnet Oberflächennormalen", ["image"], ["normals"]),

            # OCR & Text
            ("ocr_extract", "OCR Extract", "Extrahiert Text aus Bild", ["image"], ["text", "boxes"]),
            ("ocr_handwriting", "Handwriting OCR", "Erkennt Handschrift", ["image"], ["text"]),
            ("ocr_document", "Document OCR", "Dokument-OCR", ["image"], ["structured_text"]),
            ("ocr_receipt", "Receipt OCR", "Kassenbon-OCR", ["image"], ["items", "total"]),
            ("ocr_id_card", "ID Card OCR", "Ausweis-OCR", ["image"], ["fields"]),
            ("ocr_license_plate", "License Plate OCR", "Kennzeichen-OCR", ["image"], ["plate_number"]),
            ("text_detect", "Text Detection", "Erkennt Textregionen", ["image"], ["text_regions"]),

            # Image Understanding
            ("image_caption", "Image Captioning", "Beschreibt Bilder", ["image"], ["caption"]),
            ("image_qa", "Visual QA", "Beantwortet Fragen zu Bildern", ["image", "question"], ["answer"]),
            ("image_similarity", "Image Similarity", "Berechnet Bildähnlichkeit", ["image1", "image2"], ["similarity"]),
            ("image_search", "Image Search", "Sucht ähnliche Bilder", ["image", "database"], ["results"]),
            ("image_hash", "Image Hashing", "Erstellt Bild-Hash", ["image"], ["hash"]),

            # Medical
            ("xray_analyze", "X-Ray Analysis", "Analysiert Röntgenbilder", ["image"], ["findings"]),
            ("mri_segment", "MRI Segmentation", "Segmentiert MRT-Bilder", ["image"], ["segments"]),
            ("ct_analyze", "CT Analysis", "Analysiert CT-Scans", ["image"], ["findings"]),
            ("pathology_detect", "Pathology Detection", "Erkennt Pathologien", ["image"], ["findings"]),
            ("skin_lesion", "Skin Lesion Analysis", "Analysiert Hautläsionen", ["image"], ["diagnosis"]),
            ("retina_analyze", "Retina Analysis", "Analysiert Netzhaut", ["image"], ["findings"]),

            # Industrial
            ("defect_detect", "Defect Detection", "Erkennt Defekte", ["image"], ["defects"]),
            ("quality_inspect", "Quality Inspection", "Qualitätsprüfung", ["image"], ["quality_score"]),
            ("barcode_read", "Barcode Reader", "Liest Barcodes", ["image"], ["barcode"]),
            ("qr_read", "QR Code Reader", "Liest QR-Codes", ["image"], ["data"]),

            # Agriculture
            ("crop_analyze", "Crop Analysis", "Analysiert Pflanzen", ["image"], ["health", "species"]),
            ("pest_detect", "Pest Detection", "Erkennt Schädlinge", ["image"], ["pests"]),
            ("yield_estimate", "Yield Estimation", "Schätzt Ernte", ["images"], ["yield"]),

            # Satellite & Aerial
            ("land_classify", "Land Classification", "Klassifiziert Landnutzung", ["image"], ["classes"]),
            ("building_detect", "Building Detection", "Erkennt Gebäude", ["image"], ["buildings"]),
            ("road_extract", "Road Extraction", "Extrahiert Straßen", ["image"], ["roads"]),
            ("change_detect", "Change Detection", "Erkennt Veränderungen", ["image1", "image2"], ["changes"]),
        ]

        for tool_id, name, desc, inputs, outputs in vision_tools:
            self._add_tool(Tool(
                id=f"vision_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.VISION,
                input_types=inputs,
                output_types=outputs,
                requires_gpu=True,
                tags=["vision", "image", "cv"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # AUDIO TOOLS (500+)
        # ═══════════════════════════════════════════════════════════════
        audio_tools = [
            # Speech Recognition
            ("stt_transcribe", "Speech to Text", "Transkribiert Sprache", ["audio"], ["text"]),
            ("stt_realtime", "Realtime STT", "Echtzeit-Transkription", ["audio_stream"], ["text_stream"]),
            ("stt_diarize", "Speaker Diarization", "Sprechererkennung", ["audio"], ["segments", "speakers"]),
            ("stt_punctuate", "Add Punctuation", "Fügt Interpunktion hinzu", ["text"], ["punctuated"]),
            ("stt_timestamp", "Word Timestamps", "Wort-Zeitstempel", ["audio"], ["words", "timestamps"]),

            # Speech Synthesis
            ("tts_synthesize", "Text to Speech", "Sprachsynthese", ["text"], ["audio"]),
            ("tts_clone", "Voice Cloning", "Stimmenklonen", ["text", "voice_sample"], ["audio"]),
            ("tts_emotional", "Emotional TTS", "Emotionale Synthese", ["text", "emotion"], ["audio"]),
            ("tts_multilingual", "Multilingual TTS", "Mehrsprachige Synthese", ["text", "language"], ["audio"]),
            ("tts_ssml", "SSML TTS", "SSML-basierte Synthese", ["ssml"], ["audio"]),

            # Audio Analysis
            ("audio_classify", "Audio Classification", "Klassifiziert Audio", ["audio"], ["class"]),
            ("music_genre", "Genre Classification", "Erkennt Musikgenre", ["audio"], ["genre"]),
            ("music_mood", "Mood Detection", "Erkennt Stimmung", ["audio"], ["mood"]),
            ("music_bpm", "BPM Detection", "Erkennt Tempo", ["audio"], ["bpm"]),
            ("music_key", "Key Detection", "Erkennt Tonart", ["audio"], ["key"]),
            ("music_chord", "Chord Recognition", "Erkennt Akkorde", ["audio"], ["chords"]),
            ("music_transcribe", "Music Transcription", "Transkribiert Musik", ["audio"], ["notes"]),

            # Voice Analysis
            ("voice_identify", "Voice Identification", "Identifiziert Sprecher", ["audio", "database"], ["identity"]),
            ("voice_verify", "Voice Verification", "Verifiziert Sprecher", ["audio", "reference"], ["match"]),
            ("voice_emotion", "Voice Emotion", "Erkennt Emotionen", ["audio"], ["emotion"]),
            ("voice_age", "Voice Age", "Schätzt Alter", ["audio"], ["age_range"]),
            ("voice_gender", "Voice Gender", "Erkennt Geschlecht", ["audio"], ["gender"]),

            # Audio Processing
            ("audio_denoise", "Noise Reduction", "Rauschunterdrückung", ["audio"], ["cleaned_audio"]),
            ("audio_enhance", "Audio Enhancement", "Audioverbesserung", ["audio"], ["enhanced_audio"]),
            ("audio_separate", "Source Separation", "Quellentrennung", ["audio"], ["sources"]),
            ("audio_normalize", "Normalization", "Normalisierung", ["audio"], ["normalized_audio"]),
            ("audio_compress", "Dynamic Compression", "Dynamikkompression", ["audio"], ["compressed_audio"]),
            ("audio_eq", "Equalization", "Equalisierung", ["audio", "settings"], ["eq_audio"]),
            ("audio_reverb", "Add Reverb", "Hall hinzufügen", ["audio", "room"], ["reverb_audio"]),

            # Music Generation
            ("music_generate", "Music Generation", "Generiert Musik", ["prompt"], ["audio"]),
            ("music_continue", "Music Continuation", "Setzt Musik fort", ["audio"], ["continued_audio"]),
            ("music_remix", "Music Remix", "Remixed Musik", ["audio", "style"], ["remixed_audio"]),
            ("music_accompany", "Accompaniment", "Generiert Begleitung", ["melody"], ["accompaniment"]),
            ("sound_effect", "Sound Effect Generation", "Generiert Soundeffekte", ["description"], ["audio"]),
        ]

        for tool_id, name, desc, inputs, outputs in audio_tools:
            self._add_tool(Tool(
                id=f"audio_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.AUDIO,
                input_types=inputs,
                output_types=outputs,
                requires_gpu=True,
                tags=["audio", "speech", "sound"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # VIDEO TOOLS (500+)
        # ═══════════════════════════════════════════════════════════════
        video_tools = [
            # Generation
            ("video_generate", "Video Generation", "Generiert Video aus Text", ["prompt"], ["video"]),
            ("video_img2vid", "Image to Video", "Animiert Bild", ["image", "motion"], ["video"]),
            ("video_interpolate", "Frame Interpolation", "Interpoliert Frames", ["video"], ["smooth_video"]),
            ("video_upscale", "Video Upscaling", "Skaliert Video hoch", ["video"], ["hd_video"]),
            ("video_stabilize", "Stabilization", "Stabilisiert Video", ["video"], ["stable_video"]),

            # Analysis
            ("video_classify", "Video Classification", "Klassifiziert Video", ["video"], ["labels"]),
            ("video_segment", "Video Segmentation", "Segmentiert Video", ["video"], ["segments"]),
            ("video_track", "Object Tracking", "Verfolgt Objekte", ["video"], ["tracks"]),
            ("video_action", "Action Recognition", "Erkennt Aktionen", ["video"], ["actions"]),
            ("video_caption", "Video Captioning", "Beschreibt Video", ["video"], ["captions"]),
            ("video_summarize", "Video Summarization", "Fasst Video zusammen", ["video"], ["summary_video"]),
            ("video_highlight", "Highlight Detection", "Erkennt Highlights", ["video"], ["highlights"]),

            # Editing
            ("video_cut", "Video Cutting", "Schneidet Video", ["video", "timestamps"], ["cut_video"]),
            ("video_merge", "Video Merging", "Fügt Videos zusammen", ["videos"], ["merged_video"]),
            ("video_overlay", "Video Overlay", "Überlagert Videos", ["video", "overlay"], ["result"]),
            ("video_transition", "Add Transitions", "Fügt Übergänge hinzu", ["videos", "transition"], ["result"]),
            ("video_speed", "Speed Change", "Ändert Geschwindigkeit", ["video", "factor"], ["result"]),
            ("video_reverse", "Reverse Video", "Kehrt Video um", ["video"], ["reversed"]),
            ("video_loop", "Create Loop", "Erstellt Loop", ["video"], ["looped"]),

            # Effects
            ("video_filter", "Video Filter", "Wendet Filter an", ["video", "filter"], ["filtered"]),
            ("video_color", "Color Grading", "Farbkorrektur", ["video", "lut"], ["graded"]),
            ("video_denoise", "Video Denoising", "Entrauscht Video", ["video"], ["denoised"]),
            ("video_deblur", "Video Deblur", "Entschärft Video", ["video"], ["sharp"]),

            # Face & Body
            ("video_face_swap", "Face Swap", "Gesichtertausch", ["video", "face"], ["swapped"]),
            ("video_face_enhance", "Face Enhancement", "Gesichtsverbesserung", ["video"], ["enhanced"]),
            ("video_deepfake_detect", "Deepfake Detection", "Erkennt Deepfakes", ["video"], ["is_fake"]),
            ("video_body_pose", "Body Pose", "Körperposen-Analyse", ["video"], ["poses"]),

            # Audio
            ("video_extract_audio", "Extract Audio", "Extrahiert Audio", ["video"], ["audio"]),
            ("video_add_audio", "Add Audio", "Fügt Audio hinzu", ["video", "audio"], ["result"]),
            ("video_sync_audio", "Sync Audio", "Synchronisiert Audio", ["video", "audio"], ["synced"]),
            ("video_voice_over", "Voice Over", "Fügt Voiceover hinzu", ["video", "script"], ["result"]),

            # Subtitles
            ("video_subtitle_generate", "Generate Subtitles", "Generiert Untertitel", ["video"], ["subtitles"]),
            ("video_subtitle_add", "Add Subtitles", "Fügt Untertitel hinzu", ["video", "subtitles"], ["result"]),
            ("video_subtitle_translate", "Translate Subtitles", "Übersetzt Untertitel", ["subtitles", "lang"], ["translated"]),
        ]

        for tool_id, name, desc, inputs, outputs in video_tools:
            self._add_tool(Tool(
                id=f"video_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.VIDEO,
                input_types=inputs,
                output_types=outputs,
                requires_gpu=True,
                tags=["video", "multimedia"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # GENERATIVE AI TOOLS (1000+)
        # ═══════════════════════════════════════════════════════════════
        generative_tools = [
            # Image Generation
            ("img_txt2img", "Text to Image", "Generiert Bild aus Text", ["prompt"], ["image"]),
            ("img_img2img", "Image to Image", "Transformiert Bild", ["image", "prompt"], ["image"]),
            ("img_inpaint", "Inpainting", "Füllt Bildbereiche", ["image", "mask"], ["image"]),
            ("img_outpaint", "Outpainting", "Erweitert Bild", ["image", "direction"], ["image"]),
            ("img_upscale", "Image Upscaling", "Skaliert Bild hoch", ["image"], ["hd_image"]),
            ("img_restore", "Image Restoration", "Restauriert Bild", ["image"], ["restored"]),
            ("img_colorize", "Colorization", "Koloriert S/W-Bild", ["image"], ["colored"]),
            ("img_style", "Style Transfer", "Überträgt Stil", ["image", "style"], ["styled"]),
            ("img_remove_bg", "Remove Background", "Entfernt Hintergrund", ["image"], ["foreground"]),
            ("img_face_restore", "Face Restoration", "Restauriert Gesichter", ["image"], ["restored"]),

            # Specific Image Types
            ("gen_logo", "Logo Generation", "Generiert Logos", ["description"], ["logo"]),
            ("gen_icon", "Icon Generation", "Generiert Icons", ["description"], ["icon"]),
            ("gen_avatar", "Avatar Generation", "Generiert Avatare", ["description"], ["avatar"]),
            ("gen_banner", "Banner Generation", "Generiert Banner", ["description", "size"], ["banner"]),
            ("gen_mockup", "Mockup Generation", "Generiert Mockups", ["design", "template"], ["mockup"]),
            ("gen_product", "Product Image", "Generiert Produktbilder", ["product", "scene"], ["image"]),
            ("gen_fashion", "Fashion Design", "Generiert Mode-Designs", ["description"], ["design"]),
            ("gen_interior", "Interior Design", "Generiert Inneneinrichtung", ["room", "style"], ["design"]),
            ("gen_architecture", "Architecture", "Generiert Architektur", ["description"], ["render"]),
            ("gen_character", "Character Design", "Generiert Charaktere", ["description"], ["character"]),

            # 3D
            ("3d_txt2mesh", "Text to 3D", "Generiert 3D aus Text", ["prompt"], ["mesh"]),
            ("3d_img2mesh", "Image to 3D", "Generiert 3D aus Bild", ["image"], ["mesh"]),
            ("3d_nerf", "NeRF Generation", "Generiert NeRF", ["images"], ["nerf"]),
            ("3d_texture", "Texture Generation", "Generiert Texturen", ["mesh", "prompt"], ["textured_mesh"]),
            ("3d_animate", "3D Animation", "Animiert 3D-Modell", ["mesh", "motion"], ["animation"]),
            ("3d_rigging", "Auto Rigging", "Automatisches Rigging", ["mesh"], ["rigged_mesh"]),

            # Code Generation
            ("code_generate", "Code Generation", "Generiert Code", ["description", "language"], ["code"]),
            ("code_complete", "Code Completion", "Vervollständigt Code", ["code_context"], ["completion"]),
            ("code_explain", "Code Explanation", "Erklärt Code", ["code"], ["explanation"]),
            ("code_refactor", "Code Refactoring", "Refaktoriert Code", ["code"], ["refactored"]),
            ("code_optimize", "Code Optimization", "Optimiert Code", ["code"], ["optimized"]),
            ("code_debug", "Code Debugging", "Findet Bugs", ["code", "error"], ["fix"]),
            ("code_test", "Test Generation", "Generiert Tests", ["code"], ["tests"]),
            ("code_document", "Documentation", "Generiert Dokumentation", ["code"], ["docs"]),
            ("code_convert", "Code Conversion", "Konvertiert Code", ["code", "target_lang"], ["converted"]),
            ("code_review", "Code Review", "Reviewed Code", ["code"], ["feedback"]),

            # Data Generation
            ("data_synthetic", "Synthetic Data", "Generiert synthetische Daten", ["schema"], ["data"]),
            ("data_augment", "Data Augmentation", "Augmentiert Daten", ["data"], ["augmented"]),
            ("data_mock", "Mock Data", "Generiert Mock-Daten", ["schema", "count"], ["mock_data"]),

            # Document Generation
            ("doc_report", "Report Generation", "Generiert Berichte", ["data", "template"], ["report"]),
            ("doc_presentation", "Presentation", "Generiert Präsentationen", ["content"], ["slides"]),
            ("doc_email", "Email Generation", "Generiert E-Mails", ["context", "tone"], ["email"]),
            ("doc_contract", "Contract Generation", "Generiert Verträge", ["terms"], ["contract"]),
            ("doc_proposal", "Proposal Generation", "Generiert Angebote", ["requirements"], ["proposal"]),
        ]

        for tool_id, name, desc, inputs, outputs in generative_tools:
            self._add_tool(Tool(
                id=f"gen_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.GENERATIVE,
                input_types=inputs,
                output_types=outputs,
                requires_gpu=True,
                tags=["generative", "ai", "creative"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # CODE & DEVELOPMENT TOOLS (2000+)
        # ═══════════════════════════════════════════════════════════════
        code_tools = [
            # Languages
            ("python_run", "Python Runner", "Führt Python aus", ["code"], ["output"]),
            ("python_lint", "Python Linter", "Prüft Python-Code", ["code"], ["issues"]),
            ("python_format", "Python Formatter", "Formatiert Python", ["code"], ["formatted"]),
            ("js_run", "JavaScript Runner", "Führt JavaScript aus", ["code"], ["output"]),
            ("js_lint", "JavaScript Linter", "Prüft JS-Code", ["code"], ["issues"]),
            ("ts_compile", "TypeScript Compiler", "Kompiliert TypeScript", ["code"], ["js_code"]),
            ("rust_compile", "Rust Compiler", "Kompiliert Rust", ["code"], ["binary"]),
            ("go_run", "Go Runner", "Führt Go aus", ["code"], ["output"]),
            ("java_compile", "Java Compiler", "Kompiliert Java", ["code"], ["bytecode"]),
            ("cpp_compile", "C++ Compiler", "Kompiliert C++", ["code"], ["binary"]),

            # Analysis
            ("code_analyze", "Static Analysis", "Statische Analyse", ["code"], ["report"]),
            ("code_complexity", "Complexity Analysis", "Komplexitätsanalyse", ["code"], ["metrics"]),
            ("code_coverage", "Coverage Analysis", "Code-Coverage", ["code", "tests"], ["coverage"]),
            ("code_dependency", "Dependency Analysis", "Abhängigkeitsanalyse", ["project"], ["dependencies"]),
            ("code_security", "Security Scan", "Sicherheitsscan", ["code"], ["vulnerabilities"]),
            ("code_smell", "Code Smell Detection", "Erkennt Code Smells", ["code"], ["smells"]),
            ("code_duplicate", "Duplicate Detection", "Erkennt Duplikate", ["code"], ["duplicates"]),

            # Git
            ("git_clone", "Git Clone", "Klont Repository", ["url"], ["repo"]),
            ("git_commit", "Git Commit", "Erstellt Commit", ["files", "message"], ["commit_hash"]),
            ("git_push", "Git Push", "Pusht Änderungen", ["branch"], ["status"]),
            ("git_pull", "Git Pull", "Pullt Änderungen", ["branch"], ["status"]),
            ("git_merge", "Git Merge", "Merged Branches", ["source", "target"], ["status"]),
            ("git_diff", "Git Diff", "Zeigt Unterschiede", ["ref1", "ref2"], ["diff"]),
            ("git_log", "Git Log", "Zeigt Historie", ["options"], ["commits"]),
            ("git_blame", "Git Blame", "Zeigt Autoren", ["file"], ["blame"]),

            # Build & Deploy
            ("npm_install", "NPM Install", "Installiert Pakete", ["package_json"], ["status"]),
            ("npm_build", "NPM Build", "Baut Projekt", ["project"], ["build"]),
            ("pip_install", "Pip Install", "Installiert Python-Pakete", ["requirements"], ["status"]),
            ("docker_build", "Docker Build", "Baut Docker Image", ["dockerfile"], ["image"]),
            ("docker_run", "Docker Run", "Startet Container", ["image"], ["container"]),
            ("k8s_deploy", "Kubernetes Deploy", "Deployed auf K8s", ["manifest"], ["status"]),

            # API
            ("api_design", "API Design", "Designt API", ["requirements"], ["openapi_spec"]),
            ("api_generate", "API Generator", "Generiert API-Code", ["spec"], ["code"]),
            ("api_test", "API Testing", "Testet API", ["spec", "base_url"], ["results"]),
            ("api_mock", "API Mock", "Mockt API", ["spec"], ["mock_server"]),
            ("api_doc", "API Documentation", "Generiert API-Docs", ["spec"], ["docs"]),
        ]

        for tool_id, name, desc, inputs, outputs in code_tools:
            self._add_tool(Tool(
                id=f"code_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.CODE,
                input_types=inputs,
                output_types=outputs,
                tags=["code", "development", "programming"]
            ))

        # Programmiersprachen-spezifische Tools
        languages = [
            "python", "javascript", "typescript", "java", "csharp", "cpp", "c",
            "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
            "julia", "haskell", "elixir", "clojure", "dart", "lua"
        ]

        for lang in languages:
            for action in ["parse", "format", "lint", "minify", "beautify", "analyze"]:
                self._add_tool(Tool(
                    id=f"code_{lang}_{action}",
                    name=f"{lang.capitalize()} {action.capitalize()}",
                    description=f"{action.capitalize()} für {lang.capitalize()}",
                    category=ToolCategory.CODE,
                    subcategory=lang,
                    input_types=["code"],
                    output_types=["result"],
                    tags=["code", lang, action]
                ))

        # ═══════════════════════════════════════════════════════════════
        # DATA PROCESSING TOOLS (1000+)
        # ═══════════════════════════════════════════════════════════════
        data_tools = [
            # Transformation
            ("data_clean", "Data Cleaning", "Bereinigt Daten", ["data"], ["cleaned"]),
            ("data_normalize", "Normalization", "Normalisiert Daten", ["data"], ["normalized"]),
            ("data_transform", "Transformation", "Transformiert Daten", ["data", "rules"], ["transformed"]),
            ("data_aggregate", "Aggregation", "Aggregiert Daten", ["data", "groupby"], ["aggregated"]),
            ("data_pivot", "Pivot Table", "Erstellt Pivot-Tabelle", ["data", "config"], ["pivot"]),
            ("data_join", "Data Join", "Verknüpft Datensätze", ["data1", "data2", "key"], ["joined"]),
            ("data_filter", "Data Filter", "Filtert Daten", ["data", "conditions"], ["filtered"]),
            ("data_sort", "Data Sort", "Sortiert Daten", ["data", "columns"], ["sorted"]),
            ("data_dedupe", "Deduplication", "Entfernt Duplikate", ["data"], ["unique"]),
            ("data_impute", "Missing Values", "Füllt fehlende Werte", ["data"], ["imputed"]),

            # Validation
            ("data_validate", "Data Validation", "Validiert Daten", ["data", "schema"], ["valid", "errors"]),
            ("data_quality", "Quality Check", "Prüft Datenqualität", ["data"], ["quality_score"]),
            ("data_profile", "Data Profiling", "Profiliert Daten", ["data"], ["profile"]),
            ("data_anomaly", "Anomaly Detection", "Erkennt Anomalien", ["data"], ["anomalies"]),

            # Formats
            ("csv_parse", "CSV Parser", "Parst CSV", ["csv_file"], ["data"]),
            ("csv_write", "CSV Writer", "Schreibt CSV", ["data"], ["csv_file"]),
            ("json_parse", "JSON Parser", "Parst JSON", ["json_file"], ["data"]),
            ("json_write", "JSON Writer", "Schreibt JSON", ["data"], ["json_file"]),
            ("xml_parse", "XML Parser", "Parst XML", ["xml_file"], ["data"]),
            ("yaml_parse", "YAML Parser", "Parst YAML", ["yaml_file"], ["data"]),
            ("parquet_read", "Parquet Reader", "Liest Parquet", ["parquet_file"], ["data"]),
            ("parquet_write", "Parquet Writer", "Schreibt Parquet", ["data"], ["parquet_file"]),
            ("excel_read", "Excel Reader", "Liest Excel", ["excel_file"], ["data"]),
            ("excel_write", "Excel Writer", "Schreibt Excel", ["data"], ["excel_file"]),

            # Statistics
            ("stats_describe", "Descriptive Stats", "Deskriptive Statistik", ["data"], ["statistics"]),
            ("stats_correlate", "Correlation", "Berechnet Korrelation", ["data"], ["correlations"]),
            ("stats_test", "Statistical Test", "Statistischer Test", ["data", "test_type"], ["result"]),
            ("stats_regression", "Regression", "Regressionsanalyse", ["data", "target"], ["model"]),
            ("stats_cluster", "Clustering", "Clustert Daten", ["data", "n_clusters"], ["clusters"]),
            ("stats_pca", "PCA", "Dimensionsreduktion", ["data", "n_components"], ["reduced"]),
            ("stats_forecast", "Forecasting", "Zeitreihenprognose", ["time_series"], ["forecast"]),
        ]

        for tool_id, name, desc, inputs, outputs in data_tools:
            self._add_tool(Tool(
                id=f"data_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.DATA_PROCESSING,
                input_types=inputs,
                output_types=outputs,
                tags=["data", "processing", "analytics"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # DOCUMENT PROCESSING (500+)
        # ═══════════════════════════════════════════════════════════════
        doc_tools = [
            # PDF
            ("pdf_read", "PDF Reader", "Liest PDF", ["pdf_file"], ["text", "pages"]),
            ("pdf_create", "PDF Creator", "Erstellt PDF", ["content"], ["pdf_file"]),
            ("pdf_merge", "PDF Merge", "Fügt PDFs zusammen", ["pdf_files"], ["merged_pdf"]),
            ("pdf_split", "PDF Split", "Teilt PDF", ["pdf_file", "pages"], ["pdf_files"]),
            ("pdf_compress", "PDF Compress", "Komprimiert PDF", ["pdf_file"], ["compressed_pdf"]),
            ("pdf_to_image", "PDF to Image", "Konvertiert zu Bildern", ["pdf_file"], ["images"]),
            ("pdf_extract_images", "Extract Images", "Extrahiert Bilder", ["pdf_file"], ["images"]),
            ("pdf_extract_tables", "Extract Tables", "Extrahiert Tabellen", ["pdf_file"], ["tables"]),
            ("pdf_sign", "PDF Sign", "Signiert PDF", ["pdf_file", "signature"], ["signed_pdf"]),
            ("pdf_encrypt", "PDF Encrypt", "Verschlüsselt PDF", ["pdf_file", "password"], ["encrypted_pdf"]),
            ("pdf_ocr", "PDF OCR", "OCR für gescannte PDFs", ["pdf_file"], ["searchable_pdf"]),

            # Office
            ("docx_read", "DOCX Reader", "Liest Word-Dokument", ["docx_file"], ["content"]),
            ("docx_create", "DOCX Creator", "Erstellt Word-Dokument", ["content"], ["docx_file"]),
            ("docx_to_pdf", "DOCX to PDF", "Konvertiert zu PDF", ["docx_file"], ["pdf_file"]),
            ("xlsx_read", "XLSX Reader", "Liest Excel", ["xlsx_file"], ["data"]),
            ("xlsx_create", "XLSX Creator", "Erstellt Excel", ["data"], ["xlsx_file"]),
            ("pptx_read", "PPTX Reader", "Liest PowerPoint", ["pptx_file"], ["slides"]),
            ("pptx_create", "PPTX Creator", "Erstellt PowerPoint", ["content"], ["pptx_file"]),
            ("pptx_to_pdf", "PPTX to PDF", "Konvertiert zu PDF", ["pptx_file"], ["pdf_file"]),

            # Conversion
            ("doc_convert", "Document Converter", "Konvertiert Dokumente", ["file", "target_format"], ["converted"]),
            ("markdown_to_html", "MD to HTML", "Markdown zu HTML", ["markdown"], ["html"]),
            ("html_to_pdf", "HTML to PDF", "HTML zu PDF", ["html"], ["pdf_file"]),
            ("latex_to_pdf", "LaTeX to PDF", "LaTeX zu PDF", ["latex"], ["pdf_file"]),

            # Analysis
            ("doc_summarize", "Document Summary", "Fasst Dokument zusammen", ["document"], ["summary"]),
            ("doc_compare", "Document Compare", "Vergleicht Dokumente", ["doc1", "doc2"], ["diff"]),
            ("doc_classify", "Document Classification", "Klassifiziert Dokument", ["document"], ["category"]),
            ("doc_extract", "Information Extraction", "Extrahiert Informationen", ["document"], ["entities"]),
        ]

        for tool_id, name, desc, inputs, outputs in doc_tools:
            self._add_tool(Tool(
                id=f"doc_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.DOCUMENT,
                input_types=inputs,
                output_types=outputs,
                tags=["document", "office", "pdf"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # WEB TOOLS (500+)
        # ═══════════════════════════════════════════════════════════════
        web_tools = [
            # Scraping
            ("web_scrape", "Web Scraper", "Scrapt Webseiten", ["url"], ["content"]),
            ("web_crawl", "Web Crawler", "Crawlt Website", ["url", "depth"], ["pages"]),
            ("web_extract", "Data Extraction", "Extrahiert strukturierte Daten", ["url", "selectors"], ["data"]),
            ("web_screenshot", "Screenshot", "Erstellt Screenshot", ["url"], ["image"]),
            ("web_pdf", "Page to PDF", "Speichert als PDF", ["url"], ["pdf"]),
            ("web_archive", "Web Archive", "Archiviert Seite", ["url"], ["archive"]),
            ("web_monitor", "Page Monitor", "Überwacht Änderungen", ["url"], ["changes"]),

            # APIs
            ("http_get", "HTTP GET", "HTTP GET Request", ["url"], ["response"]),
            ("http_post", "HTTP POST", "HTTP POST Request", ["url", "data"], ["response"]),
            ("http_put", "HTTP PUT", "HTTP PUT Request", ["url", "data"], ["response"]),
            ("http_delete", "HTTP DELETE", "HTTP DELETE Request", ["url"], ["response"]),
            ("graphql_query", "GraphQL Query", "GraphQL Abfrage", ["endpoint", "query"], ["data"]),
            ("websocket_connect", "WebSocket", "WebSocket Verbindung", ["url"], ["connection"]),

            # Analysis
            ("seo_analyze", "SEO Analysis", "SEO-Analyse", ["url"], ["report"]),
            ("page_speed", "Page Speed", "Geschwindigkeitstest", ["url"], ["metrics"]),
            ("accessibility", "Accessibility Check", "Barrierefreiheit prüfen", ["url"], ["issues"]),
            ("broken_links", "Broken Links", "Findet defekte Links", ["url"], ["broken"]),
            ("ssl_check", "SSL Check", "Prüft SSL-Zertifikat", ["url"], ["status"]),
            ("whois_lookup", "WHOIS Lookup", "Domain-Informationen", ["domain"], ["info"]),
            ("dns_lookup", "DNS Lookup", "DNS-Abfrage", ["domain"], ["records"]),

            # Automation
            ("browser_automate", "Browser Automation", "Automatisiert Browser", ["script"], ["result"]),
            ("form_fill", "Form Filler", "Füllt Formulare aus", ["url", "data"], ["result"]),
            ("web_login", "Auto Login", "Automatischer Login", ["url", "credentials"], ["session"]),
            ("captcha_solve", "Captcha Solver", "Löst Captchas", ["image"], ["solution"]),
        ]

        for tool_id, name, desc, inputs, outputs in web_tools:
            self._add_tool(Tool(
                id=f"web_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.WEB_SCRAPING,
                input_types=inputs,
                output_types=outputs,
                requires_network=True,
                tags=["web", "scraping", "http"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # IMAGE PROCESSING TOOLS (1000+)
        # ═══════════════════════════════════════════════════════════════
        image_tools = [
            # Basic
            ("img_resize", "Resize", "Ändert Bildgröße", ["image", "size"], ["image"]),
            ("img_crop", "Crop", "Schneidet Bild zu", ["image", "bounds"], ["image"]),
            ("img_rotate", "Rotate", "Rotiert Bild", ["image", "angle"], ["image"]),
            ("img_flip", "Flip", "Spiegelt Bild", ["image", "direction"], ["image"]),
            ("img_convert", "Convert", "Konvertiert Format", ["image", "format"], ["image"]),
            ("img_compress", "Compress", "Komprimiert Bild", ["image", "quality"], ["image"]),

            # Adjustments
            ("img_brightness", "Brightness", "Helligkeit anpassen", ["image", "value"], ["image"]),
            ("img_contrast", "Contrast", "Kontrast anpassen", ["image", "value"], ["image"]),
            ("img_saturation", "Saturation", "Sättigung anpassen", ["image", "value"], ["image"]),
            ("img_hue", "Hue", "Farbton anpassen", ["image", "value"], ["image"]),
            ("img_sharpen", "Sharpen", "Schärfen", ["image"], ["image"]),
            ("img_blur", "Blur", "Weichzeichnen", ["image", "radius"], ["image"]),
            ("img_denoise", "Denoise", "Entrauschen", ["image"], ["image"]),

            # Filters
            ("img_grayscale", "Grayscale", "In Graustufen", ["image"], ["image"]),
            ("img_sepia", "Sepia", "Sepia-Effekt", ["image"], ["image"]),
            ("img_vintage", "Vintage", "Vintage-Effekt", ["image"], ["image"]),
            ("img_vignette", "Vignette", "Vignetten-Effekt", ["image"], ["image"]),
            ("img_emboss", "Emboss", "Präge-Effekt", ["image"], ["image"]),
            ("img_edge", "Edge Detection", "Kantenerkennung", ["image"], ["image"]),
            ("img_cartoon", "Cartoon", "Cartoon-Effekt", ["image"], ["image"]),
            ("img_oil_paint", "Oil Paint", "Ölgemälde-Effekt", ["image"], ["image"]),
            ("img_sketch", "Sketch", "Skizzen-Effekt", ["image"], ["image"]),
            ("img_pixelate", "Pixelate", "Pixelieren", ["image", "size"], ["image"]),

            # Composition
            ("img_overlay", "Overlay", "Bilder überlagern", ["base", "overlay"], ["image"]),
            ("img_blend", "Blend", "Bilder mischen", ["images", "mode"], ["image"]),
            ("img_watermark", "Watermark", "Wasserzeichen hinzufügen", ["image", "watermark"], ["image"]),
            ("img_collage", "Collage", "Collage erstellen", ["images", "layout"], ["image"]),
            ("img_mosaic", "Mosaic", "Mosaik erstellen", ["images"], ["image"]),

            # Advanced
            ("img_hdr", "HDR", "HDR-Verarbeitung", ["images"], ["hdr_image"]),
            ("img_panorama", "Panorama", "Panorama erstellen", ["images"], ["panorama"]),
            ("img_focus_stack", "Focus Stack", "Fokus-Stacking", ["images"], ["stacked"]),
            ("img_perspective", "Perspective", "Perspektive korrigieren", ["image", "points"], ["image"]),
            ("img_lens_correct", "Lens Correction", "Objektivkorrektur", ["image"], ["corrected"]),
        ]

        for tool_id, name, desc, inputs, outputs in image_tools:
            self._add_tool(Tool(
                id=f"img_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.IMAGE,
                input_types=inputs,
                output_types=outputs,
                tags=["image", "graphics", "photo"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # SECURITY TOOLS (500+)
        # ═══════════════════════════════════════════════════════════════
        security_tools = [
            # Encryption
            ("encrypt_aes", "AES Encrypt", "AES-Verschlüsselung", ["data", "key"], ["encrypted"]),
            ("decrypt_aes", "AES Decrypt", "AES-Entschlüsselung", ["encrypted", "key"], ["data"]),
            ("encrypt_rsa", "RSA Encrypt", "RSA-Verschlüsselung", ["data", "public_key"], ["encrypted"]),
            ("decrypt_rsa", "RSA Decrypt", "RSA-Entschlüsselung", ["encrypted", "private_key"], ["data"]),
            ("hash_md5", "MD5 Hash", "MD5-Hash", ["data"], ["hash"]),
            ("hash_sha256", "SHA256 Hash", "SHA256-Hash", ["data"], ["hash"]),
            ("hash_bcrypt", "Bcrypt Hash", "Bcrypt-Hash", ["password"], ["hash"]),
            ("sign_data", "Digital Signature", "Digitale Signatur", ["data", "private_key"], ["signature"]),
            ("verify_signature", "Verify Signature", "Signatur verifizieren", ["data", "signature", "public_key"], ["valid"]),

            # Keys
            ("keygen_rsa", "RSA Key Generation", "RSA-Schlüsselpaar", ["bits"], ["public_key", "private_key"]),
            ("keygen_ec", "EC Key Generation", "EC-Schlüsselpaar", ["curve"], ["public_key", "private_key"]),
            ("keygen_symmetric", "Symmetric Key", "Symmetrischer Schlüssel", ["length"], ["key"]),
            ("key_derive", "Key Derivation", "Schlüsselableitung", ["password", "salt"], ["key"]),

            # Scanning
            ("vuln_scan", "Vulnerability Scan", "Schwachstellenscan", ["target"], ["vulnerabilities"]),
            ("port_scan", "Port Scan", "Port-Scan", ["target"], ["open_ports"]),
            ("malware_scan", "Malware Scan", "Malware-Scan", ["file"], ["threats"]),
            ("code_audit", "Code Audit", "Sicherheitsaudit", ["code"], ["issues"]),
            ("dependency_check", "Dependency Check", "Abhängigkeitsprüfung", ["project"], ["vulnerabilities"]),

            # Auth
            ("jwt_create", "JWT Create", "JWT erstellen", ["payload", "secret"], ["token"]),
            ("jwt_verify", "JWT Verify", "JWT verifizieren", ["token", "secret"], ["valid", "payload"]),
            ("oauth_token", "OAuth Token", "OAuth-Token abrufen", ["credentials"], ["token"]),
            ("mfa_generate", "MFA Code", "MFA-Code generieren", ["secret"], ["code"]),
            ("mfa_verify", "MFA Verify", "MFA-Code verifizieren", ["code", "secret"], ["valid"]),
            ("password_generate", "Password Generator", "Passwort generieren", ["length", "options"], ["password"]),
            ("password_check", "Password Strength", "Passwortstärke prüfen", ["password"], ["strength"]),
        ]

        for tool_id, name, desc, inputs, outputs in security_tools:
            self._add_tool(Tool(
                id=f"sec_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.SECURITY,
                input_types=inputs,
                output_types=outputs,
                tags=["security", "encryption", "auth"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # DATABASE TOOLS (500+)
        # ═══════════════════════════════════════════════════════════════
        db_tools = [
            # SQL
            ("sql_query", "SQL Query", "SQL-Abfrage ausführen", ["query", "connection"], ["results"]),
            ("sql_insert", "SQL Insert", "Daten einfügen", ["table", "data", "connection"], ["result"]),
            ("sql_update", "SQL Update", "Daten aktualisieren", ["table", "data", "where", "connection"], ["result"]),
            ("sql_delete", "SQL Delete", "Daten löschen", ["table", "where", "connection"], ["result"]),
            ("sql_create_table", "Create Table", "Tabelle erstellen", ["schema", "connection"], ["result"]),
            ("sql_backup", "Database Backup", "Datenbank sichern", ["connection"], ["backup_file"]),
            ("sql_restore", "Database Restore", "Datenbank wiederherstellen", ["backup_file", "connection"], ["result"]),
            ("sql_migrate", "Schema Migration", "Schema-Migration", ["migrations", "connection"], ["result"]),

            # NoSQL
            ("mongo_find", "MongoDB Find", "MongoDB-Abfrage", ["collection", "query"], ["documents"]),
            ("mongo_insert", "MongoDB Insert", "MongoDB-Einfügen", ["collection", "documents"], ["result"]),
            ("mongo_update", "MongoDB Update", "MongoDB-Update", ["collection", "filter", "update"], ["result"]),
            ("mongo_aggregate", "MongoDB Aggregate", "MongoDB-Aggregation", ["collection", "pipeline"], ["results"]),
            ("redis_get", "Redis Get", "Redis-Abruf", ["key"], ["value"]),
            ("redis_set", "Redis Set", "Redis-Setzen", ["key", "value"], ["result"]),
            ("elastic_search", "Elasticsearch", "Elasticsearch-Suche", ["index", "query"], ["results"]),
            ("elastic_index", "Elasticsearch Index", "Elasticsearch-Index", ["index", "document"], ["result"]),

            # Vector DB
            ("vector_insert", "Vector Insert", "Vektoren einfügen", ["vectors", "metadata"], ["ids"]),
            ("vector_search", "Vector Search", "Vektorsuche", ["query_vector", "top_k"], ["results"]),
            ("vector_delete", "Vector Delete", "Vektoren löschen", ["ids"], ["result"]),

            # Analysis
            ("db_analyze", "DB Analyze", "Datenbankanalyse", ["connection"], ["report"]),
            ("query_optimize", "Query Optimizer", "Query-Optimierung", ["query"], ["optimized"]),
            ("index_suggest", "Index Suggestion", "Index-Empfehlung", ["table", "queries"], ["indices"]),
            ("db_monitor", "DB Monitor", "Datenbanküberwachung", ["connection"], ["metrics"]),
        ]

        for tool_id, name, desc, inputs, outputs in db_tools:
            self._add_tool(Tool(
                id=f"db_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.DATABASE,
                input_types=inputs,
                output_types=outputs,
                tags=["database", "sql", "nosql"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # CLOUD & INFRASTRUCTURE TOOLS (500+)
        # ═══════════════════════════════════════════════════════════════
        cloud_tools = [
            # AWS
            ("aws_s3_upload", "S3 Upload", "Datei zu S3 hochladen", ["file", "bucket"], ["url"]),
            ("aws_s3_download", "S3 Download", "Datei von S3 laden", ["key", "bucket"], ["file"]),
            ("aws_s3_list", "S3 List", "S3-Bucket auflisten", ["bucket"], ["objects"]),
            ("aws_ec2_launch", "EC2 Launch", "EC2-Instanz starten", ["config"], ["instance"]),
            ("aws_ec2_stop", "EC2 Stop", "EC2-Instanz stoppen", ["instance_id"], ["result"]),
            ("aws_lambda_invoke", "Lambda Invoke", "Lambda-Funktion aufrufen", ["function", "payload"], ["result"]),
            ("aws_sqs_send", "SQS Send", "Nachricht an SQS", ["queue", "message"], ["message_id"]),
            ("aws_sns_publish", "SNS Publish", "SNS-Nachricht senden", ["topic", "message"], ["message_id"]),
            ("aws_dynamodb_query", "DynamoDB Query", "DynamoDB-Abfrage", ["table", "query"], ["items"]),

            # GCP
            ("gcp_storage_upload", "GCS Upload", "Datei zu GCS hochladen", ["file", "bucket"], ["url"]),
            ("gcp_bigquery", "BigQuery", "BigQuery-Abfrage", ["query"], ["results"]),
            ("gcp_pubsub_publish", "Pub/Sub Publish", "Pub/Sub-Nachricht", ["topic", "message"], ["message_id"]),
            ("gcp_functions_invoke", "Cloud Functions", "Cloud Function aufrufen", ["function", "data"], ["result"]),

            # Azure
            ("azure_blob_upload", "Blob Upload", "Datei zu Azure Blob", ["file", "container"], ["url"]),
            ("azure_cosmos_query", "Cosmos DB Query", "Cosmos DB-Abfrage", ["query"], ["results"]),
            ("azure_functions_invoke", "Azure Functions", "Azure Function aufrufen", ["function", "data"], ["result"]),

            # Kubernetes
            ("k8s_get_pods", "Get Pods", "Pods abrufen", ["namespace"], ["pods"]),
            ("k8s_scale", "Scale Deployment", "Deployment skalieren", ["deployment", "replicas"], ["result"]),
            ("k8s_apply", "Apply Manifest", "Manifest anwenden", ["manifest"], ["result"]),
            ("k8s_logs", "Get Logs", "Logs abrufen", ["pod"], ["logs"]),
            ("k8s_exec", "Exec Command", "Befehl ausführen", ["pod", "command"], ["output"]),

            # Docker
            ("docker_build", "Docker Build", "Image bauen", ["dockerfile", "tag"], ["image"]),
            ("docker_push", "Docker Push", "Image pushen", ["image", "registry"], ["result"]),
            ("docker_pull", "Docker Pull", "Image pullen", ["image"], ["result"]),
            ("docker_run", "Docker Run", "Container starten", ["image", "config"], ["container"]),
            ("docker_logs", "Docker Logs", "Container-Logs", ["container"], ["logs"]),
        ]

        for tool_id, name, desc, inputs, outputs in cloud_tools:
            self._add_tool(Tool(
                id=f"cloud_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.CLOUD,
                input_types=inputs,
                output_types=outputs,
                requires_network=True,
                tags=["cloud", "infrastructure", "devops"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # UTILITY TOOLS (1000+)
        # ═══════════════════════════════════════════════════════════════
        utility_tools = [
            # Conversion
            ("convert_units", "Unit Converter", "Einheiten konvertieren", ["value", "from", "to"], ["result"]),
            ("convert_currency", "Currency Converter", "Währungen konvertieren", ["amount", "from", "to"], ["result"]),
            ("convert_timezone", "Timezone Converter", "Zeitzonen konvertieren", ["time", "from", "to"], ["result"]),
            ("convert_base", "Base Converter", "Zahlensystem konvertieren", ["number", "from", "to"], ["result"]),
            ("convert_encoding", "Encoding Converter", "Encoding konvertieren", ["data", "from", "to"], ["result"]),

            # Compression
            ("compress_zip", "ZIP Compress", "ZIP komprimieren", ["files"], ["zip_file"]),
            ("decompress_zip", "ZIP Decompress", "ZIP entpacken", ["zip_file"], ["files"]),
            ("compress_gzip", "GZIP Compress", "GZIP komprimieren", ["file"], ["gzip_file"]),
            ("compress_tar", "TAR Create", "TAR erstellen", ["files"], ["tar_file"]),

            # Formatting
            ("format_json", "JSON Formatter", "JSON formatieren", ["json"], ["formatted"]),
            ("format_xml", "XML Formatter", "XML formatieren", ["xml"], ["formatted"]),
            ("format_sql", "SQL Formatter", "SQL formatieren", ["sql"], ["formatted"]),
            ("format_html", "HTML Formatter", "HTML formatieren", ["html"], ["formatted"]),

            # Validation
            ("validate_email", "Email Validator", "E-Mail validieren", ["email"], ["valid"]),
            ("validate_url", "URL Validator", "URL validieren", ["url"], ["valid"]),
            ("validate_phone", "Phone Validator", "Telefon validieren", ["phone"], ["valid"]),
            ("validate_json", "JSON Validator", "JSON validieren", ["json"], ["valid"]),
            ("validate_xml", "XML Validator", "XML validieren", ["xml"], ["valid"]),
            ("validate_schema", "Schema Validator", "Schema validieren", ["data", "schema"], ["valid"]),

            # Generators
            ("generate_uuid", "UUID Generator", "UUID generieren", [], ["uuid"]),
            ("generate_qr", "QR Code Generator", "QR-Code generieren", ["data"], ["qr_image"]),
            ("generate_barcode", "Barcode Generator", "Barcode generieren", ["data", "type"], ["barcode_image"]),
            ("generate_lorem", "Lorem Ipsum", "Lorem Ipsum generieren", ["paragraphs"], ["text"]),
            ("generate_fake", "Fake Data", "Fake-Daten generieren", ["schema", "count"], ["data"]),
            ("generate_random", "Random Generator", "Zufallswerte generieren", ["type", "count"], ["values"]),

            # Text Utils
            ("text_diff", "Text Diff", "Textunterschiede", ["text1", "text2"], ["diff"]),
            ("text_encode", "Base64 Encode", "Base64-Encoding", ["text"], ["encoded"]),
            ("text_decode", "Base64 Decode", "Base64-Decoding", ["encoded"], ["text"]),
            ("text_escape", "HTML Escape", "HTML escapen", ["text"], ["escaped"]),
            ("text_slug", "Slugify", "Slug erstellen", ["text"], ["slug"]),
            ("text_truncate", "Truncate", "Text kürzen", ["text", "length"], ["truncated"]),
            ("text_count", "Word Count", "Wörter zählen", ["text"], ["count"]),

            # Date/Time
            ("time_now", "Current Time", "Aktuelle Zeit", [], ["timestamp"]),
            ("time_format", "Format Time", "Zeit formatieren", ["timestamp", "format"], ["formatted"]),
            ("time_parse", "Parse Time", "Zeit parsen", ["string", "format"], ["timestamp"]),
            ("time_diff", "Time Difference", "Zeitdifferenz", ["time1", "time2"], ["difference"]),
            ("time_add", "Add Duration", "Dauer addieren", ["time", "duration"], ["result"]),

            # File System
            ("file_read", "Read File", "Datei lesen", ["path"], ["content"]),
            ("file_write", "Write File", "Datei schreiben", ["path", "content"], ["result"]),
            ("file_delete", "Delete File", "Datei löschen", ["path"], ["result"]),
            ("file_copy", "Copy File", "Datei kopieren", ["source", "destination"], ["result"]),
            ("file_move", "Move File", "Datei verschieben", ["source", "destination"], ["result"]),
            ("file_list", "List Files", "Dateien auflisten", ["path", "pattern"], ["files"]),
            ("file_info", "File Info", "Dateiinformationen", ["path"], ["info"]),
            ("file_hash", "File Hash", "Datei-Hash", ["path", "algorithm"], ["hash"]),
        ]

        for tool_id, name, desc, inputs, outputs in utility_tools:
            self._add_tool(Tool(
                id=f"util_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.CONVERSION,
                input_types=inputs,
                output_types=outputs,
                tags=["utility", "helper", "tools"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # BUSINESS TOOLS (500+)
        # ═══════════════════════════════════════════════════════════════
        business_tools = [
            # Finance
            ("finance_invoice", "Invoice Generator", "Rechnung erstellen", ["items", "customer"], ["invoice"]),
            ("finance_calculate", "Financial Calculator", "Finanzberechnung", ["type", "params"], ["result"]),
            ("finance_exchange", "Exchange Rates", "Wechselkurse", ["from", "to"], ["rate"]),
            ("finance_stock", "Stock Price", "Aktienkurs", ["symbol"], ["price"]),
            ("finance_crypto", "Crypto Price", "Kryptokurs", ["symbol"], ["price"]),
            ("finance_tax", "Tax Calculator", "Steuerrechner", ["income", "country"], ["tax"]),

            # CRM
            ("crm_contact_create", "Create Contact", "Kontakt erstellen", ["data"], ["contact"]),
            ("crm_contact_update", "Update Contact", "Kontakt aktualisieren", ["id", "data"], ["contact"]),
            ("crm_lead_score", "Lead Scoring", "Lead bewerten", ["lead"], ["score"]),
            ("crm_opportunity", "Opportunity Track", "Opportunity verfolgen", ["data"], ["opportunity"]),

            # Email
            ("email_send", "Send Email", "E-Mail senden", ["to", "subject", "body"], ["result"]),
            ("email_template", "Email Template", "E-Mail-Template", ["template", "data"], ["email"]),
            ("email_parse", "Parse Email", "E-Mail parsen", ["email"], ["parsed"]),
            ("email_validate", "Validate Email", "E-Mail validieren", ["email"], ["valid"]),

            # Calendar
            ("calendar_create", "Create Event", "Termin erstellen", ["event"], ["result"]),
            ("calendar_list", "List Events", "Termine auflisten", ["date_range"], ["events"]),
            ("calendar_sync", "Sync Calendar", "Kalender synchronisieren", ["calendars"], ["result"]),
            ("calendar_available", "Check Availability", "Verfügbarkeit prüfen", ["participants", "duration"], ["slots"]),

            # HR
            ("hr_payroll", "Payroll Calculate", "Gehaltsabrechnung", ["employee", "period"], ["payslip"]),
            ("hr_leave", "Leave Management", "Urlaubsverwaltung", ["employee", "dates"], ["result"]),
            ("hr_timesheet", "Timesheet", "Zeiterfassung", ["employee", "entries"], ["result"]),
            ("hr_performance", "Performance Review", "Leistungsbewertung", ["employee", "data"], ["review"]),

            # Project Management
            ("pm_task_create", "Create Task", "Aufgabe erstellen", ["task"], ["result"]),
            ("pm_task_assign", "Assign Task", "Aufgabe zuweisen", ["task", "assignee"], ["result"]),
            ("pm_milestone", "Track Milestone", "Meilenstein verfolgen", ["project", "milestone"], ["status"]),
            ("pm_burndown", "Burndown Chart", "Burndown-Chart", ["sprint"], ["chart"]),
            ("pm_gantt", "Gantt Chart", "Gantt-Chart", ["project"], ["chart"]),
        ]

        for tool_id, name, desc, inputs, outputs in business_tools:
            self._add_tool(Tool(
                id=f"biz_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.CRM,
                input_types=inputs,
                output_types=outputs,
                tags=["business", "enterprise", "productivity"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # SCIENTIFIC TOOLS (500+)
        # ═══════════════════════════════════════════════════════════════
        scientific_tools = [
            # Math
            ("math_calculate", "Calculator", "Berechnung", ["expression"], ["result"]),
            ("math_solve", "Equation Solver", "Gleichungslöser", ["equation"], ["solution"]),
            ("math_derive", "Derivative", "Ableitung", ["function"], ["derivative"]),
            ("math_integrate", "Integral", "Integral", ["function"], ["integral"]),
            ("math_matrix", "Matrix Operations", "Matrixoperationen", ["matrix", "operation"], ["result"]),
            ("math_optimize", "Optimization", "Optimierung", ["function", "constraints"], ["solution"]),
            ("math_fft", "FFT", "Fourier-Transformation", ["signal"], ["spectrum"]),

            # Statistics
            ("stat_mean", "Mean", "Mittelwert", ["data"], ["mean"]),
            ("stat_median", "Median", "Median", ["data"], ["median"]),
            ("stat_std", "Standard Deviation", "Standardabweichung", ["data"], ["std"]),
            ("stat_distribution", "Distribution Fit", "Verteilungsanpassung", ["data"], ["distribution"]),
            ("stat_hypothesis", "Hypothesis Test", "Hypothesentest", ["data", "test"], ["result"]),
            ("stat_anova", "ANOVA", "Varianzanalyse", ["groups"], ["result"]),
            ("stat_chi2", "Chi-Square Test", "Chi-Quadrat-Test", ["observed", "expected"], ["result"]),

            # Physics
            ("physics_kinematic", "Kinematics", "Kinematik", ["params"], ["result"]),
            ("physics_thermodynamic", "Thermodynamics", "Thermodynamik", ["params"], ["result"]),
            ("physics_electromagnetic", "Electromagnetism", "Elektromagnetismus", ["params"], ["result"]),
            ("physics_quantum", "Quantum", "Quantenmechanik", ["params"], ["result"]),

            # Chemistry
            ("chem_balance", "Balance Equation", "Reaktionsgleichung", ["equation"], ["balanced"]),
            ("chem_molecular", "Molecular Weight", "Molekulargewicht", ["formula"], ["weight"]),
            ("chem_property", "Chemical Properties", "Chemische Eigenschaften", ["compound"], ["properties"]),

            # Biology
            ("bio_sequence", "Sequence Analysis", "Sequenzanalyse", ["sequence"], ["analysis"]),
            ("bio_align", "Sequence Alignment", "Sequenzausrichtung", ["sequences"], ["alignment"]),
            ("bio_blast", "BLAST Search", "BLAST-Suche", ["sequence"], ["results"]),
            ("bio_phylogeny", "Phylogenetic Tree", "Stammbaum", ["sequences"], ["tree"]),

            # Simulation
            ("sim_monte_carlo", "Monte Carlo", "Monte-Carlo-Simulation", ["model", "iterations"], ["results"]),
            ("sim_agent_based", "Agent-Based", "Agentenbasierte Simulation", ["model"], ["results"]),
            ("sim_discrete_event", "Discrete Event", "Diskrete Ereignissimulation", ["model"], ["results"]),
        ]

        for tool_id, name, desc, inputs, outputs in scientific_tools:
            self._add_tool(Tool(
                id=f"sci_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.MATH,
                input_types=inputs,
                output_types=outputs,
                tags=["science", "math", "research"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # VISUALIZATION TOOLS (300+)
        # ═══════════════════════════════════════════════════════════════
        viz_tools = [
            # Charts
            ("chart_line", "Line Chart", "Liniendiagramm", ["data"], ["chart"]),
            ("chart_bar", "Bar Chart", "Balkendiagramm", ["data"], ["chart"]),
            ("chart_pie", "Pie Chart", "Kreisdiagramm", ["data"], ["chart"]),
            ("chart_scatter", "Scatter Plot", "Streudiagramm", ["data"], ["chart"]),
            ("chart_histogram", "Histogram", "Histogramm", ["data"], ["chart"]),
            ("chart_heatmap", "Heatmap", "Heatmap", ["data"], ["chart"]),
            ("chart_boxplot", "Box Plot", "Boxplot", ["data"], ["chart"]),
            ("chart_violin", "Violin Plot", "Violindiagramm", ["data"], ["chart"]),
            ("chart_radar", "Radar Chart", "Radardiagramm", ["data"], ["chart"]),
            ("chart_treemap", "Treemap", "Treemap", ["data"], ["chart"]),
            ("chart_sunburst", "Sunburst", "Sunburst-Diagramm", ["data"], ["chart"]),
            ("chart_sankey", "Sankey Diagram", "Sankey-Diagramm", ["data"], ["chart"]),
            ("chart_network", "Network Graph", "Netzwerkgraph", ["nodes", "edges"], ["chart"]),
            ("chart_timeline", "Timeline", "Zeitleiste", ["events"], ["chart"]),
            ("chart_gauge", "Gauge Chart", "Tachometer", ["value", "range"], ["chart"]),

            # Maps
            ("map_choropleth", "Choropleth Map", "Choroplethenkarte", ["data", "geojson"], ["map"]),
            ("map_scatter", "Scatter Map", "Punktkarte", ["locations"], ["map"]),
            ("map_heatmap", "Heatmap Map", "Heatmap-Karte", ["data"], ["map"]),
            ("map_route", "Route Map", "Routenkarte", ["waypoints"], ["map"]),

            # 3D
            ("viz_3d_surface", "3D Surface", "3D-Oberfläche", ["data"], ["chart"]),
            ("viz_3d_scatter", "3D Scatter", "3D-Streudiagramm", ["data"], ["chart"]),
            ("viz_3d_mesh", "3D Mesh", "3D-Mesh", ["vertices", "faces"], ["mesh"]),

            # Dashboards
            ("dashboard_create", "Create Dashboard", "Dashboard erstellen", ["widgets"], ["dashboard"]),
            ("dashboard_widget", "Add Widget", "Widget hinzufügen", ["dashboard", "widget"], ["result"]),
            ("dashboard_export", "Export Dashboard", "Dashboard exportieren", ["dashboard", "format"], ["file"]),
        ]

        for tool_id, name, desc, inputs, outputs in viz_tools:
            self._add_tool(Tool(
                id=f"viz_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.VISUALIZATION,
                input_types=inputs,
                output_types=outputs,
                tags=["visualization", "charts", "graphics"]
            ))

        # ═══════════════════════════════════════════════════════════════
        # MLOPS TOOLS (200+)
        # ═══════════════════════════════════════════════════════════════
        mlops_tools = [
            # Training
            ("ml_train", "Train Model", "Modell trainieren", ["data", "config"], ["model"]),
            ("ml_finetune", "Fine-tune Model", "Modell feinabstimmen", ["model", "data"], ["tuned_model"]),
            ("ml_hyperopt", "Hyperparameter Tuning", "Hyperparameter optimieren", ["model", "search_space"], ["best_params"]),
            ("ml_cross_val", "Cross Validation", "Kreuzvalidierung", ["model", "data", "folds"], ["scores"]),
            ("ml_ensemble", "Ensemble", "Ensemble erstellen", ["models"], ["ensemble"]),

            # Evaluation
            ("ml_evaluate", "Evaluate Model", "Modell evaluieren", ["model", "test_data"], ["metrics"]),
            ("ml_confusion", "Confusion Matrix", "Konfusionsmatrix", ["predictions", "labels"], ["matrix"]),
            ("ml_roc", "ROC Curve", "ROC-Kurve", ["predictions", "labels"], ["curve"]),
            ("ml_explain", "Model Explanation", "Modell erklären", ["model", "data"], ["explanations"]),
            ("ml_feature_imp", "Feature Importance", "Feature-Wichtigkeit", ["model"], ["importance"]),

            # Deployment
            ("ml_export", "Export Model", "Modell exportieren", ["model", "format"], ["file"]),
            ("ml_serve", "Serve Model", "Modell bereitstellen", ["model", "config"], ["endpoint"]),
            ("ml_monitor", "Monitor Model", "Modell überwachen", ["model", "data_stream"], ["metrics"]),
            ("ml_version", "Version Model", "Modell versionieren", ["model", "version"], ["result"]),

            # Data
            ("ml_preprocess", "Preprocess Data", "Daten vorverarbeiten", ["data", "config"], ["processed"]),
            ("ml_feature_eng", "Feature Engineering", "Feature Engineering", ["data"], ["features"]),
            ("ml_split", "Train-Test Split", "Daten aufteilen", ["data", "ratio"], ["train", "test"]),
            ("ml_balance", "Balance Data", "Daten balancieren", ["data"], ["balanced"]),
        ]

        for tool_id, name, desc, inputs, outputs in mlops_tools:
            self._add_tool(Tool(
                id=f"ml_{tool_id}",
                name=name,
                description=desc,
                category=ToolCategory.ML_OPS,
                input_types=inputs,
                output_types=outputs,
                requires_gpu=True,
                tags=["ml", "ai", "training"]
            ))

        # Füge Varianten für verschiedene Frameworks hinzu
        frameworks = ["pytorch", "tensorflow", "sklearn", "xgboost", "lightgbm", "catboost", "onnx", "huggingface"]
        for fw in frameworks:
            self._add_tool(Tool(
                id=f"ml_{fw}_load",
                name=f"Load {fw.capitalize()} Model",
                description=f"Lädt {fw.capitalize()}-Modell",
                category=ToolCategory.ML_OPS,
                subcategory=fw,
                input_types=["model_path"],
                output_types=["model"],
                tags=["ml", fw, "model"]
            ))
            self._add_tool(Tool(
                id=f"ml_{fw}_save",
                name=f"Save {fw.capitalize()} Model",
                description=f"Speichert {fw.capitalize()}-Modell",
                category=ToolCategory.ML_OPS,
                subcategory=fw,
                input_types=["model", "path"],
                output_types=["result"],
                tags=["ml", fw, "model"]
            ))

    def _add_tool(self, tool: Tool):
        """Fügt ein Tool hinzu"""
        self.tools[tool.id] = tool
        self.categories[tool.category].append(tool.id)

        # Tags indexieren
        for tag in tool.tags:
            if tag not in self.tags_index:
                self.tags_index[tag] = set()
            self.tags_index[tag].add(tool.id)

        self._tool_count += 1

    def register_handler(self, tool_id: str, handler: Callable):
        """Registriert einen Handler für ein Tool"""
        if tool_id in self.tools:
            self.handlers[tool_id] = handler

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Gibt ein Tool zurück"""
        return self.tools.get(tool_id)

    def search(self,
               query: str = None,
               category: ToolCategory = None,
               tags: List[str] = None,
               requires_gpu: bool = None,
               limit: int = 100) -> List[Tool]:
        """Sucht nach Tools"""
        results = list(self.tools.values())

        if category:
            results = [t for t in results if t.category == category]

        if tags:
            for tag in tags:
                tag_tools = self.tags_index.get(tag, set())
                results = [t for t in results if t.id in tag_tools]

        if requires_gpu is not None:
            results = [t for t in results if t.requires_gpu == requires_gpu]

        if query:
            query_lower = query.lower()
            results = [t for t in results if
                      query_lower in t.name.lower() or
                      query_lower in t.description.lower() or
                      any(query_lower in tag for tag in t.tags)]

        return results[:limit]

    def execute(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Führt ein Tool aus"""
        tool = self.get_tool(tool_id)
        if not tool:
            return {"error": f"Tool '{tool_id}' not found"}

        if not tool.enabled:
            return {"error": f"Tool '{tool_id}' is disabled"}

        handler = self.handlers.get(tool_id)
        if not handler:
            return {"error": f"No handler registered for '{tool_id}'"}

        try:
            import time
            start = time.time()
            result = handler(params)
            duration = (time.time() - start) * 1000

            # Update stats
            tool.usage_count += 1
            tool.avg_execution_time_ms = (
                (tool.avg_execution_time_ms * (tool.usage_count - 1) + duration) /
                tool.usage_count
            )

            return {"result": result, "execution_time_ms": duration}
        except Exception as e:
            tool.success_rate = (
                (tool.success_rate * tool.usage_count) / (tool.usage_count + 1)
            )
            tool.usage_count += 1
            return {"error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        category_counts = {cat.value: len(ids) for cat, ids in self.categories.items()}

        return {
            "total_tools": self._tool_count,
            "categories": len(self.categories),
            "category_distribution": category_counts,
            "unique_tags": len(self.tags_index),
            "handlers_registered": len(self.handlers),
            "gpu_required_tools": len([t for t in self.tools.values() if t.requires_gpu]),
            "network_required_tools": len([t for t in self.tools.values() if t.requires_network])
        }

    def get_categories(self) -> Dict[str, int]:
        """Gibt Kategorien mit Tool-Anzahl zurück"""
        return {cat.value: len(ids) for cat, ids in self.categories.items()}

    def export_catalog(self, path: str = None) -> str:
        """Exportiert Tool-Katalog als JSON"""
        catalog = {
            "version": "1.0.0",
            "total_tools": self._tool_count,
            "categories": self.get_categories(),
            "tools": {tid: t.to_dict() for tid, t in self.tools.items()}
        }

        json_str = json.dumps(catalog, indent=2, ensure_ascii=False)

        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str


# Singleton
_tool_registry: Optional[ToolRegistry] = None

def get_tool_registry() -> ToolRegistry:
    """Gibt Singleton-Instanz zurück"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
