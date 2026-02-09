#!/usr/bin/env python3
"""
SCIO - Browser Automation (Selenium-based)

Automatisiert Browser-Interaktionen fuer Web-Scraping,
Testing und RPA-Aufgaben.
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BrowserType(str, Enum):
    """Unterstuetzte Browser"""
    CHROME = "chrome"
    FIREFOX = "firefox"
    EDGE = "edge"
    HEADLESS_CHROME = "headless_chrome"
    HEADLESS_FIREFOX = "headless_firefox"


@dataclass
class WebElement:
    """Repraesentation eines Web-Elements"""
    tag_name: str
    text: str
    attributes: Dict[str, str] = field(default_factory=dict)
    location: Dict[str, int] = field(default_factory=dict)
    size: Dict[str, int] = field(default_factory=dict)
    is_displayed: bool = True
    is_enabled: bool = True


@dataclass
class PageInfo:
    """Informationen ueber die aktuelle Seite"""
    url: str
    title: str
    source: str = ""
    cookies: List[Dict] = field(default_factory=list)
    screenshot_path: Optional[str] = None


class BrowserAutomation:
    """
    Browser-Automatisierung mit Selenium

    Ermoeglicht:
    - Navigieren zu URLs
    - Formulare ausfuellen
    - Elemente klicken
    - Screenshots erstellen
    - JavaScript ausfuehren
    - Cookies verwalten
    """

    def __init__(self, browser_type: BrowserType = BrowserType.HEADLESS_CHROME):
        self.browser_type = browser_type
        self._driver = None
        self._initialized = False
        self._screenshot_dir = Path("data/screenshots")
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _init_driver(self):
        """Initialisiert den WebDriver"""
        if self._initialized:
            return

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.firefox.options import Options as FirefoxOptions

            if self.browser_type in [BrowserType.CHROME, BrowserType.HEADLESS_CHROME]:
                options = ChromeOptions()
                if self.browser_type == BrowserType.HEADLESS_CHROME:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                self._driver = webdriver.Chrome(options=options)

            elif self.browser_type in [BrowserType.FIREFOX, BrowserType.HEADLESS_FIREFOX]:
                options = FirefoxOptions()
                if self.browser_type == BrowserType.HEADLESS_FIREFOX:
                    options.add_argument("--headless")
                self._driver = webdriver.Firefox(options=options)

            elif self.browser_type == BrowserType.EDGE:
                from selenium.webdriver.edge.options import Options as EdgeOptions
                options = EdgeOptions()
                self._driver = webdriver.Edge(options=options)

            self._initialized = True
            print(f"[BROWSER] {self.browser_type.value} initialisiert")

        except ImportError:
            print("[ERROR] Selenium nicht installiert: pip install selenium")
            raise
        except Exception as e:
            print(f"[ERROR] Browser-Initialisierung fehlgeschlagen: {e}")
            raise

    def navigate(self, url: str, wait_seconds: float = 2.0) -> PageInfo:
        """Navigiert zu einer URL"""
        self._init_driver()
        self._driver.get(url)
        time.sleep(wait_seconds)

        return PageInfo(
            url=self._driver.current_url,
            title=self._driver.title,
            source=self._driver.page_source,
            cookies=self._driver.get_cookies()
        )

    def find_element(self, selector: str, by: str = "css") -> Optional[WebElement]:
        """Findet ein Element auf der Seite"""
        self._init_driver()

        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import NoSuchElementException

        by_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "name": By.NAME,
            "class": By.CLASS_NAME,
            "tag": By.TAG_NAME,
            "link_text": By.LINK_TEXT,
        }

        try:
            element = self._driver.find_element(by_map.get(by, By.CSS_SELECTOR), selector)
            return WebElement(
                tag_name=element.tag_name,
                text=element.text,
                attributes={attr: element.get_attribute(attr) for attr in ["id", "class", "href", "src", "value"]},
                location=element.location,
                size=element.size,
                is_displayed=element.is_displayed(),
                is_enabled=element.is_enabled()
            )
        except NoSuchElementException:
            return None

    def find_elements(self, selector: str, by: str = "css") -> List[WebElement]:
        """Findet mehrere Elemente"""
        self._init_driver()

        from selenium.webdriver.common.by import By

        by_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "class": By.CLASS_NAME,
            "tag": By.TAG_NAME,
        }

        elements = self._driver.find_elements(by_map.get(by, By.CSS_SELECTOR), selector)

        return [
            WebElement(
                tag_name=el.tag_name,
                text=el.text,
                attributes={attr: el.get_attribute(attr) for attr in ["id", "class", "href", "src"]},
                is_displayed=el.is_displayed(),
                is_enabled=el.is_enabled()
            )
            for el in elements
        ]

    def click(self, selector: str, by: str = "css", wait_seconds: float = 0.5) -> bool:
        """Klickt auf ein Element"""
        self._init_driver()

        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException

        by_map = {"css": By.CSS_SELECTOR, "xpath": By.XPATH, "id": By.ID}

        try:
            element = self._driver.find_element(by_map.get(by, By.CSS_SELECTOR), selector)
            element.click()
            time.sleep(wait_seconds)
            return True
        except (NoSuchElementException, ElementClickInterceptedException) as e:
            print(f"[ERROR] Click fehlgeschlagen: {e}")
            return False

    def fill_input(self, selector: str, text: str, by: str = "css", clear_first: bool = True) -> bool:
        """Fuellt ein Eingabefeld aus"""
        self._init_driver()

        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import NoSuchElementException

        by_map = {"css": By.CSS_SELECTOR, "xpath": By.XPATH, "id": By.ID, "name": By.NAME}

        try:
            element = self._driver.find_element(by_map.get(by, By.CSS_SELECTOR), selector)
            if clear_first:
                element.clear()
            element.send_keys(text)
            return True
        except NoSuchElementException as e:
            print(f"[ERROR] Input nicht gefunden: {e}")
            return False

    def submit_form(self, form_selector: str = "form") -> bool:
        """Sendet ein Formular ab"""
        self._init_driver()

        from selenium.webdriver.common.by import By

        try:
            form = self._driver.find_element(By.CSS_SELECTOR, form_selector)
            form.submit()
            return True
        except Exception as e:
            print(f"[ERROR] Form-Submit fehlgeschlagen: {e}")
            return False

    def execute_script(self, script: str, *args) -> Any:
        """Fuehrt JavaScript aus"""
        self._init_driver()
        return self._driver.execute_script(script, *args)

    def screenshot(self, name: str = "screenshot") -> str:
        """Erstellt einen Screenshot"""
        self._init_driver()

        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.png"
        filepath = self._screenshot_dir / filename

        self._driver.save_screenshot(str(filepath))
        return str(filepath)

    def scroll_to(self, y: int = 0, x: int = 0):
        """Scrollt zu einer Position"""
        self._init_driver()
        self._driver.execute_script(f"window.scrollTo({x}, {y})")

    def scroll_to_element(self, selector: str):
        """Scrollt zu einem Element"""
        self._init_driver()
        from selenium.webdriver.common.by import By

        try:
            element = self._driver.find_element(By.CSS_SELECTOR, selector)
            self._driver.execute_script("arguments[0].scrollIntoView();", element)
        except Exception:
            pass

    def wait_for_element(self, selector: str, timeout: int = 10, by: str = "css") -> bool:
        """Wartet auf ein Element"""
        self._init_driver()

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        by_map = {"css": By.CSS_SELECTOR, "xpath": By.XPATH, "id": By.ID}

        try:
            WebDriverWait(self._driver, timeout).until(
                EC.presence_of_element_located((by_map.get(by, By.CSS_SELECTOR), selector))
            )
            return True
        except Exception:
            return False

    def get_cookies(self) -> List[Dict]:
        """Gibt alle Cookies zurueck"""
        self._init_driver()
        return self._driver.get_cookies()

    def set_cookie(self, name: str, value: str, domain: str = None):
        """Setzt ein Cookie"""
        self._init_driver()
        cookie = {"name": name, "value": value}
        if domain:
            cookie["domain"] = domain
        self._driver.add_cookie(cookie)

    def delete_cookies(self):
        """Loescht alle Cookies"""
        self._init_driver()
        self._driver.delete_all_cookies()

    def switch_to_frame(self, frame_selector: str):
        """Wechselt zu einem iFrame"""
        self._init_driver()
        from selenium.webdriver.common.by import By

        frame = self._driver.find_element(By.CSS_SELECTOR, frame_selector)
        self._driver.switch_to.frame(frame)

    def switch_to_default(self):
        """Zurueck zum Hauptfenster"""
        self._init_driver()
        self._driver.switch_to.default_content()

    def get_page_source(self) -> str:
        """Gibt den Seitenquelltext zurueck"""
        self._init_driver()
        return self._driver.page_source

    def get_current_url(self) -> str:
        """Gibt die aktuelle URL zurueck"""
        self._init_driver()
        return self._driver.current_url

    def go_back(self):
        """Geht zurueck"""
        self._init_driver()
        self._driver.back()

    def go_forward(self):
        """Geht vorwaerts"""
        self._init_driver()
        self._driver.forward()

    def refresh(self):
        """Aktualisiert die Seite"""
        self._init_driver()
        self._driver.refresh()

    def close(self):
        """Schliesst den Browser"""
        if self._driver:
            self._driver.quit()
            self._driver = None
            self._initialized = False
            print("[BROWSER] Browser geschlossen")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FormFiller:
    """Automatisches Formular-Ausfuellen"""

    def __init__(self, browser: BrowserAutomation):
        self.browser = browser

    def fill_form(self, url: str, form_data: Dict[str, str], submit: bool = True) -> bool:
        """
        Fuellt ein Formular automatisch aus

        Args:
            url: URL der Seite
            form_data: Dict mit {selector: value}
            submit: Formular absenden?
        """
        try:
            self.browser.navigate(url)

            for selector, value in form_data.items():
                self.browser.fill_input(selector, value)

            if submit:
                self.browser.submit_form()

            return True
        except Exception as e:
            print(f"[ERROR] Formular ausfuellen fehlgeschlagen: {e}")
            return False


class WebAutomationWorkflow:
    """Workflow-basierte Web-Automatisierung"""

    def __init__(self, browser: BrowserAutomation = None):
        self.browser = browser or BrowserAutomation()
        self.steps: List[Dict] = []
        self.results: List[Any] = []

    def add_step(self, action: str, **kwargs) -> 'WebAutomationWorkflow':
        """Fuegt einen Schritt hinzu"""
        self.steps.append({"action": action, **kwargs})
        return self

    def navigate(self, url: str) -> 'WebAutomationWorkflow':
        return self.add_step("navigate", url=url)

    def click(self, selector: str) -> 'WebAutomationWorkflow':
        return self.add_step("click", selector=selector)

    def fill(self, selector: str, text: str) -> 'WebAutomationWorkflow':
        return self.add_step("fill", selector=selector, text=text)

    def wait(self, seconds: float) -> 'WebAutomationWorkflow':
        return self.add_step("wait", seconds=seconds)

    def screenshot(self, name: str) -> 'WebAutomationWorkflow':
        return self.add_step("screenshot", name=name)

    def execute(self) -> List[Any]:
        """Fuehrt den Workflow aus"""
        self.results = []

        for step in self.steps:
            action = step["action"]

            if action == "navigate":
                result = self.browser.navigate(step["url"])
            elif action == "click":
                result = self.browser.click(step["selector"])
            elif action == "fill":
                result = self.browser.fill_input(step["selector"], step["text"])
            elif action == "wait":
                time.sleep(step["seconds"])
                result = True
            elif action == "screenshot":
                result = self.browser.screenshot(step["name"])
            else:
                result = None

            self.results.append(result)

        return self.results

    def close(self):
        self.browser.close()


# Singleton
_browser: Optional[BrowserAutomation] = None


def get_browser_automation() -> BrowserAutomation:
    """Gibt Browser-Automation Singleton zurueck"""
    global _browser
    if _browser is None:
        _browser = BrowserAutomation()
    return _browser
