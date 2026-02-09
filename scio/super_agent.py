#!/usr/bin/env python3
"""
SCIO - Universal Digital Super-Agent

Vereint ALLE Faehigkeiten von SCIO in einem maechtigen Agenten:
- Software Development
- Cybersecurity
- AI/ML
- Data Analysis
- Knowledge Processing
- Web Scraping
- NLP
- Decision Making
- Strategic Planning
- Automation
- Hardware/IoT
- API Communication
- Enterprise Integration
- Social Media
- Trading & Finance
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class CapabilityDomain(str, Enum):
    """Faehigkeits-Domaenen"""
    SOFTWARE_DEV = "software_development"
    CYBERSECURITY = "cybersecurity"
    AI_ML = "artificial_intelligence"
    DATA_ANALYSIS = "data_analysis"
    KNOWLEDGE = "knowledge_processing"
    WEB = "web_scraping"
    NLP = "natural_language"
    DECISION = "decision_making"
    PLANNING = "strategic_planning"
    AUTOMATION = "automation"
    HARDWARE = "hardware_iot"
    API = "api_communication"
    ENTERPRISE = "enterprise"
    SOCIAL = "social_media"
    FINANCE = "finance_trading"


@dataclass
class AgentCapability:
    """Eine Faehigkeit des Agenten"""
    name: str
    domain: CapabilityDomain
    description: str
    is_available: bool = True
    requires_config: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None


@dataclass
class AgentState:
    """Zustand des Super-Agenten"""
    is_active: bool = False
    current_task: str = ""
    active_domains: List[CapabilityDomain] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    uptime_hours: float = 0.0
    tasks_completed: int = 0
    errors_count: int = 0


class UniversalSuperAgent:
    """
    SCIO Universal Digital Super-Agent

    Ein einziger Agent mit ALLEN Faehigkeiten die ein
    autonomes digitales System braucht.
    """

    def __init__(self):
        self._state = AgentState()
        self._capabilities: Dict[str, AgentCapability] = {}
        self._start_time = datetime.now()

        # Lazy-loaded Module
        self._nlp = None
        self._planner = None
        self._guardrails = None
        self._knowledge = None
        self._trading = None
        self._social = None
        self._enterprise = None
        self._iot = None
        self._browser = None

        # Alle Faehigkeiten registrieren
        self._register_capabilities()

    def _register_capabilities(self):
        """Registriert alle Faehigkeiten"""

        # Software Development
        self._add_capability("code_execution", CapabilityDomain.SOFTWARE_DEV,
                           "Python und Shell Code ausfuehren")
        self._add_capability("code_generation", CapabilityDomain.SOFTWARE_DEV,
                           "Code aus Beschreibungen generieren")
        self._add_capability("git_operations", CapabilityDomain.SOFTWARE_DEV,
                           "Git Repositories verwalten")

        # AI/ML
        self._add_capability("model_training", CapabilityDomain.AI_ML,
                           "ML Modelle trainieren")
        self._add_capability("inference", CapabilityDomain.AI_ML,
                           "KI-Inferenz durchfuehren")
        self._add_capability("embeddings", CapabilityDomain.AI_ML,
                           "Text-Embeddings erstellen")

        # Data Analysis
        self._add_capability("statistics", CapabilityDomain.DATA_ANALYSIS,
                           "Statistische Analysen")
        self._add_capability("visualization", CapabilityDomain.DATA_ANALYSIS,
                           "Datenvisualisierung")
        self._add_capability("pattern_detection", CapabilityDomain.DATA_ANALYSIS,
                           "Muster erkennen")

        # Knowledge Processing
        self._add_capability("knowledge_graph", CapabilityDomain.KNOWLEDGE,
                           "Wissensgraph verwalten")
        self._add_capability("reasoning", CapabilityDomain.KNOWLEDGE,
                           "Logisches Schlussfolgern")
        self._add_capability("semantic_search", CapabilityDomain.KNOWLEDGE,
                           "Semantische Suche")

        # Web
        self._add_capability("web_scraping", CapabilityDomain.WEB,
                           "Webseiten scrapen")
        self._add_capability("browser_automation", CapabilityDomain.WEB,
                           "Browser automatisieren")
        self._add_capability("web_search", CapabilityDomain.WEB,
                           "Web-Suche durchfuehren")

        # NLP
        self._add_capability("text_analysis", CapabilityDomain.NLP,
                           "Textanalyse (Sentiment, Entities, Keywords)")
        self._add_capability("summarization", CapabilityDomain.NLP,
                           "Textzusammenfassung")
        self._add_capability("translation", CapabilityDomain.NLP,
                           "Sprachuebersetzung")

        # Decision Making
        self._add_capability("decision_tree", CapabilityDomain.DECISION,
                           "Entscheidungsbaum-basierte Entscheidungen")
        self._add_capability("risk_assessment", CapabilityDomain.DECISION,
                           "Risikobewertung")
        self._add_capability("optimization", CapabilityDomain.DECISION,
                           "Optimierungsprobleme loesen")

        # Planning
        self._add_capability("htn_planning", CapabilityDomain.PLANNING,
                           "Hierarchische Aufgabenplanung")
        self._add_capability("workflow_orchestration", CapabilityDomain.PLANNING,
                           "Workflow-Orchestrierung")
        self._add_capability("resource_management", CapabilityDomain.PLANNING,
                           "Ressourcen-Verwaltung")

        # Automation
        self._add_capability("task_automation", CapabilityDomain.AUTOMATION,
                           "Aufgaben automatisieren")
        self._add_capability("scheduled_jobs", CapabilityDomain.AUTOMATION,
                           "Geplante Ausfuehrung")
        self._add_capability("money_making", CapabilityDomain.AUTOMATION,
                           "Automatisches Geldverdienen")

        # Hardware/IoT
        self._add_capability("iot_devices", CapabilityDomain.HARDWARE,
                           "IoT-Geraete steuern")
        self._add_capability("robotics", CapabilityDomain.HARDWARE,
                           "Roboter steuern")
        self._add_capability("smart_home", CapabilityDomain.HARDWARE,
                           "Smart Home Integration")

        # API
        self._add_capability("rest_api", CapabilityDomain.API,
                           "REST APIs aufrufen")
        self._add_capability("graphql", CapabilityDomain.API,
                           "GraphQL Queries")
        self._add_capability("webhooks", CapabilityDomain.API,
                           "Webhooks verarbeiten")

        # Enterprise
        self._add_capability("google_workspace", CapabilityDomain.ENTERPRISE,
                           "Google Workspace Integration")
        self._add_capability("slack", CapabilityDomain.ENTERPRISE,
                           "Slack Integration")
        self._add_capability("github", CapabilityDomain.ENTERPRISE,
                           "GitHub Integration")

        # Social Media
        self._add_capability("twitter", CapabilityDomain.SOCIAL,
                           "Twitter/X Integration")
        self._add_capability("reddit", CapabilityDomain.SOCIAL,
                           "Reddit Integration")
        self._add_capability("telegram", CapabilityDomain.SOCIAL,
                           "Telegram Bot")

        # Finance/Trading
        self._add_capability("crypto_trading", CapabilityDomain.FINANCE,
                           "Krypto-Handel")
        self._add_capability("market_analysis", CapabilityDomain.FINANCE,
                           "Marktanalyse")
        self._add_capability("portfolio_management", CapabilityDomain.FINANCE,
                           "Portfolio-Management")

        # Cybersecurity
        self._add_capability("security_scan", CapabilityDomain.CYBERSECURITY,
                           "Sicherheits-Scans")
        self._add_capability("vulnerability_check", CapabilityDomain.CYBERSECURITY,
                           "Schwachstellen pruefen")
        self._add_capability("content_filter", CapabilityDomain.CYBERSECURITY,
                           "Content-Filterung")

    def _add_capability(self, name: str, domain: CapabilityDomain, description: str):
        """Fuegt Faehigkeit hinzu"""
        self._capabilities[name] = AgentCapability(
            name=name,
            domain=domain,
            description=description
        )

    # === Lazy Loading der Module ===

    @property
    def nlp(self):
        """NLP-Modul"""
        if self._nlp is None:
            from scio.nlp import get_nlp
            self._nlp = get_nlp()
        return self._nlp

    @property
    def planner(self):
        """HTN Planner"""
        if self._planner is None:
            from scio.planning import get_htn_planner
            self._planner = get_htn_planner()
        return self._planner

    @property
    def guardrails(self):
        """Guardrails System"""
        if self._guardrails is None:
            from scio.safety import get_guardrails
            self._guardrails = get_guardrails()
        return self._guardrails

    @property
    def knowledge(self):
        """Knowledge Graph"""
        if self._knowledge is None:
            from scio.knowledge.graph import get_knowledge_graph
            self._knowledge = get_knowledge_graph()
        return self._knowledge

    @property
    def trading(self):
        """Trading System"""
        if self._trading is None:
            from scio.trading import TradingSystem
            self._trading = TradingSystem()
        return self._trading

    @property
    def social(self):
        """Social Media Manager"""
        if self._social is None:
            from scio.integrations import get_social_media_manager
            self._social = get_social_media_manager()
        return self._social

    @property
    def enterprise(self):
        """Enterprise Manager"""
        if self._enterprise is None:
            from scio.integrations import get_enterprise_manager
            self._enterprise = get_enterprise_manager()
        return self._enterprise

    @property
    def iot(self):
        """IoT Device Manager"""
        if self._iot is None:
            from scio.hardware.iot_robotics import get_iot_manager
            self._iot = get_iot_manager()
        return self._iot

    @property
    def browser(self):
        """Browser Automation"""
        if self._browser is None:
            from scio.web.browser_automation import get_browser_automation
            self._browser = get_browser_automation()
        return self._browser

    # === Haupt-Methoden ===

    def get_capabilities(self, domain: CapabilityDomain = None) -> List[AgentCapability]:
        """Gibt alle Faehigkeiten zurueck"""
        caps = list(self._capabilities.values())
        if domain:
            caps = [c for c in caps if c.domain == domain]
        return caps

    def get_state(self) -> AgentState:
        """Gibt aktuellen Zustand zurueck"""
        self._state.uptime_hours = (datetime.now() - self._start_time).total_seconds() / 3600
        return self._state

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analysiert Text mit NLP"""
        analysis = self.nlp.analyze(text)
        return {
            "language": analysis.language,
            "sentiment": analysis.sentiment,
            "keywords": analysis.keywords[:10],
            "entities": [{"text": e.text, "label": e.label} for e in analysis.entities],
            "summary": analysis.summary
        }

    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """Durchsucht das Web"""
        from scio.web.search import WebSearch
        search = WebSearch()
        return search.search(query, num_results)

    def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scraped eine Webseite"""
        from scio.web.scraper import WebScraper
        scraper = WebScraper()
        return scraper.scrape(url)

    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Fuehrt Code aus"""
        from scio.tools.builtin.python_executor import PythonExecutor
        executor = PythonExecutor()
        result = executor.execute(code)
        return {
            "success": result.success,
            "output": result.stdout,
            "error": result.stderr,
            "return_value": result.result
        }

    def plan_task(self, task_name: str, goal: Dict = None) -> Dict[str, Any]:
        """Plant eine Aufgabe mit HTN"""
        from scio.planning import Task, TaskType

        task = Task(name=task_name, task_type=TaskType.COMPOUND)
        self.planner.set_goal(goal or {})
        plan = self.planner.plan([task])

        return {
            "is_valid": plan.is_valid,
            "steps": [t.name for t in plan.tasks],
            "total_cost": plan.total_cost,
            "total_duration": plan.total_duration
        }

    def check_safety(self, action: str, content: str = None) -> Dict[str, Any]:
        """Prueft Sicherheit einer Aktion"""
        from scio.safety import ActionType
        result = self.guardrails.check(action, ActionType.READ, content)
        return {
            "is_valid": result.is_valid,
            "risk_level": result.risk_level.value,
            "reason": result.reason,
            "warnings": result.warnings
        }

    def get_market_analysis(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Holt Marktanalyse"""
        try:
            from scio.trading.crypto import CryptoPriceFetcher
            from scio.trading.sentiment import SentimentAggregator

            prices = CryptoPriceFetcher()
            price_data = prices.get_price(symbol.lower())

            sentiment = SentimentAggregator()
            sentiment_data = sentiment.get_overall_sentiment()

            return {
                "symbol": symbol,
                "price": price_data,
                "sentiment": sentiment_data
            }
        except Exception as e:
            return {"error": str(e)}

    def notify(self, title: str, message: str) -> Dict[str, bool]:
        """Sendet Benachrichtigung ueber alle Kanaele"""
        results = {}

        # Discord/Telegram
        try:
            results.update(self.social.notify_all(title, message))
        except Exception:
            pass

        # Slack
        try:
            if self.enterprise.slack._enabled:
                results["slack"] = self.enterprise.slack.send_message("#general", f"*{title}*\n{message}")
        except Exception:
            pass

        return results

    def earn_money(self) -> Dict[str, Any]:
        """Startet autonomes Geldverdienen"""
        try:
            from backend.automation.money_maker import get_money_maker
            money_maker = get_money_maker()
            money_maker.start()
            stats = money_maker.get_stats()
            return {
                "status": "running",
                "total_earnings": stats.total_earnings_usd,
                "hourly_rate": stats.current_hourly_rate
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_summary(self) -> Dict[str, Any]:
        """Gibt Zusammenfassung des Agenten"""
        state = self.get_state()

        # Verfuegbare Domains
        available_domains = set()
        for cap in self._capabilities.values():
            if cap.is_available:
                available_domains.add(cap.domain.value)

        return {
            "name": "SCIO Universal Super-Agent",
            "version": "1.0.0",
            "state": {
                "is_active": state.is_active,
                "uptime_hours": round(state.uptime_hours, 2),
                "tasks_completed": state.tasks_completed
            },
            "capabilities": {
                "total": len(self._capabilities),
                "domains": list(available_domains),
                "by_domain": {
                    domain.value: len([c for c in self._capabilities.values() if c.domain == domain])
                    for domain in CapabilityDomain
                }
            },
            "integrations": {
                "social_media": self.social.get_available_platforms() if self._social else [],
                "enterprise": self.enterprise.get_available_integrations() if self._enterprise else []
            }
        }


# Singleton
_super_agent: Optional[UniversalSuperAgent] = None


def get_super_agent() -> UniversalSuperAgent:
    """Gibt Super-Agent Singleton zurueck"""
    global _super_agent
    if _super_agent is None:
        _super_agent = UniversalSuperAgent()
    return _super_agent
