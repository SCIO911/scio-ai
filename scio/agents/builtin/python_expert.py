"""
SCIO Python Expert Agent

Umfassendes Python-Wissen und Code-Generierung.
"""

import ast
import inspect
import sys
from typing import Any, Optional

from pydantic import Field

from scio.agents.base import Agent, AgentConfig, AgentContext
from scio.agents.registry import register_agent
from scio.core.logging import get_logger

logger = get_logger(__name__)


# Python-Wissensbasis
PYTHON_KNOWLEDGE = {
    "builtin_types": {
        "int": "Ganzzahl - unbegrenzte Praezision. z.B. x = 42",
        "float": "Gleitkommazahl - 64-bit IEEE 754. z.B. x = 3.14",
        "str": "Unicode-String - immutable. z.B. s = 'hello'",
        "bytes": "Byte-Sequenz - immutable. z.B. b = b'hello'",
        "bool": "Boolean - True/False, Subklasse von int",
        "list": "Mutable Sequenz. z.B. lst = [1, 2, 3]",
        "tuple": "Immutable Sequenz. z.B. t = (1, 2, 3)",
        "dict": "Hash-Map/Dictionary. z.B. d = {'key': 'value'}",
        "set": "Mutable Menge eindeutiger Elemente. z.B. s = {1, 2, 3}",
        "frozenset": "Immutable Menge. z.B. fs = frozenset([1, 2, 3])",
        "None": "Singleton fuer 'kein Wert'. z.B. x = None",
        "complex": "Komplexe Zahl. z.B. c = 3+4j",
        "range": "Immutable Zahlensequenz. z.B. r = range(10)",
        "bytearray": "Mutable Byte-Sequenz. z.B. ba = bytearray(b'hello')",
        "memoryview": "Speicheransicht auf Bytes. z.B. mv = memoryview(b'hello')",
    },
    "builtin_functions": {
        "print": "Ausgabe auf stdout. print(*args, sep=' ', end='\\n', file=sys.stdout)",
        "len": "Laenge einer Sequenz. len(obj) -> int",
        "range": "Zahlensequenz. range(stop) oder range(start, stop, step)",
        "type": "Typ eines Objekts. type(obj) -> type",
        "isinstance": "Typenpruefung. isinstance(obj, class_or_tuple) -> bool",
        "str": "String-Konvertierung. str(obj) -> str",
        "int": "Integer-Konvertierung. int(x, base=10) -> int",
        "float": "Float-Konvertierung. float(x) -> float",
        "list": "Liste erstellen. list(iterable) -> list",
        "dict": "Dictionary erstellen. dict(**kwargs) oder dict(mapping)",
        "set": "Set erstellen. set(iterable) -> set",
        "tuple": "Tuple erstellen. tuple(iterable) -> tuple",
        "sorted": "Sortierte Liste. sorted(iterable, key=None, reverse=False)",
        "reversed": "Umgekehrter Iterator. reversed(sequence)",
        "enumerate": "Index-Wert-Paare. enumerate(iterable, start=0)",
        "zip": "Parallele Iteration. zip(*iterables)",
        "map": "Funktion auf Elemente anwenden. map(func, *iterables)",
        "filter": "Elemente filtern. filter(func, iterable)",
        "sum": "Summe. sum(iterable, start=0)",
        "min": "Minimum. min(iterable) oder min(a, b, ...)",
        "max": "Maximum. max(iterable) oder max(a, b, ...)",
        "abs": "Absolutwert. abs(x) -> number",
        "round": "Runden. round(number, ndigits=None)",
        "all": "Alle wahr? all(iterable) -> bool",
        "any": "Mindestens eins wahr? any(iterable) -> bool",
        "open": "Datei oeffnen. open(file, mode='r', encoding=None)",
        "input": "Benutzereingabe. input(prompt='') -> str",
        "repr": "Repr-String. repr(obj) -> str",
        "hash": "Hash-Wert. hash(obj) -> int",
        "id": "Objekt-ID. id(obj) -> int",
        "dir": "Attribute auflisten. dir(obj) -> list",
        "vars": "Objekt-Dictionary. vars(obj) -> dict",
        "getattr": "Attribut lesen. getattr(obj, name, default)",
        "setattr": "Attribut setzen. setattr(obj, name, value)",
        "hasattr": "Attribut vorhanden? hasattr(obj, name) -> bool",
        "delattr": "Attribut loeschen. delattr(obj, name)",
        "callable": "Aufrufbar? callable(obj) -> bool",
        "iter": "Iterator erstellen. iter(obj) oder iter(callable, sentinel)",
        "next": "Naechstes Element. next(iterator, default)",
        "slice": "Slice-Objekt. slice(start, stop, step)",
        "format": "Formatierung. format(value, format_spec)",
        "eval": "Python-Ausdruck auswerten. eval(expression, globals, locals)",
        "exec": "Python-Code ausfuehren. exec(code, globals, locals)",
        "compile": "Code kompilieren. compile(source, filename, mode)",
        "globals": "Globaler Namespace. globals() -> dict",
        "locals": "Lokaler Namespace. locals() -> dict",
        "ord": "Unicode-Codepoint. ord(char) -> int",
        "chr": "Zeichen aus Codepoint. chr(i) -> str",
        "bin": "Binaer-String. bin(x) -> str",
        "oct": "Oktal-String. oct(x) -> str",
        "hex": "Hexadezimal-String. hex(x) -> str",
        "pow": "Potenz. pow(base, exp, mod=None)",
        "divmod": "Division und Modulo. divmod(a, b) -> (quotient, remainder)",
        "property": "Property-Decorator. @property",
        "classmethod": "Klassenmethode. @classmethod",
        "staticmethod": "Statische Methode. @staticmethod",
        "super": "Elternklasse. super() oder super(type, obj)",
        "object": "Basis aller Klassen. class MyClass(object):",
        "issubclass": "Subklassen-Pruefung. issubclass(cls, classinfo) -> bool",
        "memoryview": "Speicheransicht. memoryview(obj)",
        "bytearray": "Byte-Array. bytearray(source, encoding, errors)",
        "bytes": "Bytes-Objekt. bytes(source, encoding, errors)",
        "ascii": "ASCII-Repr. ascii(obj) -> str",
        "breakpoint": "Debugger starten. breakpoint()",
        "help": "Hilfe anzeigen. help(obj)",
    },
    "keywords": {
        "False": "Boolean-Literal False",
        "True": "Boolean-Literal True",
        "None": "None-Literal",
        "and": "Logisches UND: a and b",
        "or": "Logisches ODER: a or b",
        "not": "Logische Negation: not a",
        "if": "Bedingte Ausfuehrung: if condition:",
        "elif": "Else-If: elif condition:",
        "else": "Else-Block: else:",
        "for": "For-Schleife: for item in iterable:",
        "while": "While-Schleife: while condition:",
        "break": "Schleife abbrechen",
        "continue": "Naechste Iteration",
        "pass": "Leerer Block (Platzhalter)",
        "def": "Funktion definieren: def func(args):",
        "return": "Wert zurueckgeben: return value",
        "yield": "Generator-Wert: yield value",
        "class": "Klasse definieren: class MyClass:",
        "try": "Exception-Handling: try:",
        "except": "Exception fangen: except Exception as e:",
        "finally": "Immer ausfuehren: finally:",
        "raise": "Exception werfen: raise Exception('msg')",
        "with": "Context-Manager: with open(f) as file:",
        "as": "Alias: import x as y, with a as b",
        "import": "Modul importieren: import module",
        "from": "Selektiver Import: from module import name",
        "global": "Globale Variable: global var",
        "nonlocal": "Aeussere Variable: nonlocal var",
        "lambda": "Anonyme Funktion: lambda x: x + 1",
        "assert": "Assertion: assert condition, message",
        "del": "Loeschen: del obj",
        "in": "Enthaltensein: x in container",
        "is": "Identitaet: a is b",
        "async": "Async-Funktion: async def func():",
        "await": "Auf Coroutine warten: await coro",
        "match": "Pattern Matching (3.10+): match value:",
        "case": "Match-Case: case pattern:",
        "type": "Type-Alias (3.12+): type Alias = Original",
    },
    "dunder_methods": {
        "__init__": "Konstruktor. def __init__(self, args):",
        "__new__": "Instanz-Erstellung. def __new__(cls, args):",
        "__del__": "Destruktor. def __del__(self):",
        "__repr__": "Repr-String. def __repr__(self) -> str:",
        "__str__": "String-Darstellung. def __str__(self) -> str:",
        "__len__": "Laenge. def __len__(self) -> int:",
        "__getitem__": "Index-Zugriff. def __getitem__(self, key):",
        "__setitem__": "Index-Zuweisung. def __setitem__(self, key, value):",
        "__delitem__": "Index-Loeschung. def __delitem__(self, key):",
        "__iter__": "Iterator. def __iter__(self):",
        "__next__": "Naechstes Element. def __next__(self):",
        "__contains__": "Enthaltensein. def __contains__(self, item) -> bool:",
        "__call__": "Aufrufbar machen. def __call__(self, args):",
        "__enter__": "Context-Manager Entry. def __enter__(self):",
        "__exit__": "Context-Manager Exit. def __exit__(self, exc_type, exc_val, exc_tb):",
        "__add__": "Addition. def __add__(self, other):",
        "__sub__": "Subtraktion. def __sub__(self, other):",
        "__mul__": "Multiplikation. def __mul__(self, other):",
        "__truediv__": "Division. def __truediv__(self, other):",
        "__floordiv__": "Ganzzahl-Division. def __floordiv__(self, other):",
        "__mod__": "Modulo. def __mod__(self, other):",
        "__pow__": "Potenz. def __pow__(self, other):",
        "__eq__": "Gleichheit. def __eq__(self, other) -> bool:",
        "__ne__": "Ungleichheit. def __ne__(self, other) -> bool:",
        "__lt__": "Kleiner als. def __lt__(self, other) -> bool:",
        "__le__": "Kleiner gleich. def __le__(self, other) -> bool:",
        "__gt__": "Groesser als. def __gt__(self, other) -> bool:",
        "__ge__": "Groesser gleich. def __ge__(self, other) -> bool:",
        "__hash__": "Hash-Wert. def __hash__(self) -> int:",
        "__bool__": "Boolean-Wert. def __bool__(self) -> bool:",
        "__getattr__": "Attribut nicht gefunden. def __getattr__(self, name):",
        "__setattr__": "Attribut setzen. def __setattr__(self, name, value):",
        "__delattr__": "Attribut loeschen. def __delattr__(self, name):",
        "__slots__": "Attribute einschraenken. __slots__ = ['attr1', 'attr2']",
        "__class__": "Klasse des Objekts. obj.__class__",
        "__dict__": "Objekt-Attribute. obj.__dict__",
        "__doc__": "Docstring. obj.__doc__",
        "__name__": "Name. obj.__name__",
        "__module__": "Modul. obj.__module__",
        "__bases__": "Basisklassen. cls.__bases__",
        "__mro__": "Method Resolution Order. cls.__mro__",
        "__subclasses__": "Subklassen. cls.__subclasses__()",
        "__annotations__": "Type-Hints. func.__annotations__",
        "__await__": "Awaitable. def __await__(self):",
        "__aiter__": "Async Iterator. async def __aiter__(self):",
        "__anext__": "Async Next. async def __anext__(self):",
        "__aenter__": "Async Context Enter. async def __aenter__(self):",
        "__aexit__": "Async Context Exit. async def __aexit__(self, ...):",
    },
    "stdlib_modules": {
        "os": "Betriebssystem-Interface. os.path, os.environ, os.getcwd()",
        "sys": "System-spezifisch. sys.path, sys.argv, sys.exit()",
        "pathlib": "Objektorientierte Pfade. Path('/path/to/file')",
        "json": "JSON Encoding/Decoding. json.dumps(), json.loads()",
        "re": "Regulaere Ausdruecke. re.match(), re.search(), re.findall()",
        "datetime": "Datum und Zeit. datetime.now(), timedelta",
        "collections": "Container-Typen. Counter, defaultdict, deque, namedtuple",
        "itertools": "Iterator-Tools. chain, cycle, repeat, combinations",
        "functools": "Funktions-Tools. partial, reduce, lru_cache, wraps",
        "typing": "Type-Hints. List, Dict, Optional, Union, Callable",
        "dataclasses": "Datenklassen. @dataclass",
        "abc": "Abstrakte Basisklassen. ABC, abstractmethod",
        "contextlib": "Context-Manager. contextmanager, suppress",
        "logging": "Logging. logging.info(), logging.error()",
        "unittest": "Unit-Tests. TestCase, assertEqual",
        "pytest": "Testing-Framework. @pytest.fixture, assert",
        "asyncio": "Async IO. async/await, asyncio.run()",
        "threading": "Threads. Thread, Lock, Event",
        "multiprocessing": "Prozesse. Process, Pool, Queue",
        "concurrent.futures": "Executor. ThreadPoolExecutor, ProcessPoolExecutor",
        "subprocess": "Subprozesse. subprocess.run(), Popen",
        "socket": "Netzwerk-Sockets. socket.socket()",
        "http": "HTTP. http.client, http.server",
        "urllib": "URL-Handling. urllib.request, urllib.parse",
        "requests": "HTTP-Client (3rd party). requests.get(), requests.post()",
        "sqlite3": "SQLite-Datenbank. sqlite3.connect()",
        "pickle": "Object Serialization. pickle.dump(), pickle.load()",
        "copy": "Kopieren. copy.copy(), copy.deepcopy()",
        "pprint": "Pretty Print. pprint.pprint()",
        "textwrap": "Text-Wrapping. textwrap.wrap(), textwrap.dedent()",
        "struct": "Binaer-Daten. struct.pack(), struct.unpack()",
        "io": "I/O-Streams. StringIO, BytesIO",
        "tempfile": "Temp-Dateien. NamedTemporaryFile, TemporaryDirectory",
        "shutil": "Datei-Operationen. shutil.copy(), shutil.rmtree()",
        "glob": "Datei-Patterns. glob.glob('*.py')",
        "fnmatch": "Filename-Matching. fnmatch.fnmatch()",
        "hashlib": "Hash-Funktionen. hashlib.sha256(), hashlib.md5()",
        "secrets": "Sichere Zufallszahlen. secrets.token_hex()",
        "random": "Zufallszahlen. random.randint(), random.choice()",
        "math": "Mathematik. math.sqrt(), math.pi, math.sin()",
        "statistics": "Statistik. statistics.mean(), statistics.stdev()",
        "decimal": "Dezimal-Arithmetik. Decimal('0.1')",
        "fractions": "Brueche. Fraction(1, 3)",
        "operator": "Operatoren als Funktionen. operator.add, operator.itemgetter",
        "argparse": "Argument-Parsing. ArgumentParser",
        "configparser": "Config-Dateien. ConfigParser",
        "csv": "CSV-Dateien. csv.reader(), csv.writer()",
        "xml": "XML-Verarbeitung. xml.etree.ElementTree",
        "html": "HTML-Utilities. html.escape(), html.parser",
        "email": "E-Mail-Handling. email.message",
        "base64": "Base64-Encoding. base64.b64encode()",
        "zlib": "Kompression. zlib.compress(), zlib.decompress()",
        "gzip": "Gzip-Kompression. gzip.open()",
        "zipfile": "ZIP-Archive. ZipFile",
        "tarfile": "TAR-Archive. TarFile",
        "enum": "Enumerations. Enum, IntEnum, Flag",
        "weakref": "Weak References. weakref.ref()",
        "gc": "Garbage Collection. gc.collect()",
        "inspect": "Introspection. inspect.signature(), inspect.getsource()",
        "dis": "Disassembler. dis.dis(func)",
        "ast": "Abstract Syntax Tree. ast.parse(), ast.dump()",
        "traceback": "Stack Traces. traceback.print_exc()",
        "warnings": "Warnungen. warnings.warn()",
        "time": "Zeit. time.time(), time.sleep()",
        "calendar": "Kalender. calendar.month()",
        "locale": "Lokalisierung. locale.setlocale()",
        "gettext": "Internationalisierung. gettext.gettext()",
        "platform": "Plattform-Info. platform.system()",
        "ctypes": "C-Interop. ctypes.cdll",
        "array": "Effiziente Arrays. array.array('i', [1, 2, 3])",
        "queue": "Thread-sichere Queues. Queue, PriorityQueue",
        "heapq": "Heap-Algorithmen. heapq.heappush(), heapq.heappop()",
        "bisect": "Binaersuche. bisect.bisect(), bisect.insort()",
    },
    "design_patterns": {
        "singleton": """
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
""",
        "factory": """
class AnimalFactory:
    @staticmethod
    def create(animal_type: str):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown: {animal_type}")
""",
        "observer": """
class Observable:
    def __init__(self):
        self._observers = []

    def subscribe(self, observer):
        self._observers.append(observer)

    def notify(self, data):
        for observer in self._observers:
            observer.update(data)
""",
        "decorator": """
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - start:.2f}s")
        return result
    return wrapper
""",
        "context_manager": """
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)
""",
        "iterator": """
class CountDown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1
""",
        "generator": """
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
""",
        "async_pattern": """
async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks)
""",
    },
    "best_practices": {
        "naming": "snake_case fuer Variablen/Funktionen, PascalCase fuer Klassen, UPPER_CASE fuer Konstanten",
        "docstrings": "Google/NumPy/Sphinx-Style Docstrings fuer alle oeffentlichen APIs",
        "type_hints": "Type-Hints fuer Funktions-Signaturen: def func(x: int) -> str:",
        "exceptions": "Spezifische Exceptions fangen, nicht bare except:",
        "context_managers": "with-Statement fuer Ressourcen (Dateien, Locks, etc.)",
        "list_comprehension": "List Comprehensions statt map/filter: [x*2 for x in lst if x > 0]",
        "generators": "Generators fuer grosse Datenmengen: (x for x in large_list)",
        "f_strings": "f-Strings fuer Formatierung: f'Name: {name}'",
        "pathlib": "pathlib.Path statt os.path fuer Pfad-Operationen",
        "dataclasses": "@dataclass fuer Datencontainer statt manueller __init__",
        "enumerate": "enumerate() statt range(len()): for i, item in enumerate(lst):",
        "zip": "zip() fuer parallele Iteration: for a, b in zip(list1, list2):",
        "unpacking": "Tuple-Unpacking: a, b = b, a",
        "walrus": "Walrus-Operator (3.8+): if (n := len(lst)) > 10:",
        "match": "Pattern Matching (3.10+) statt if/elif-Ketten",
    },
}


class PythonExpertConfig(AgentConfig):
    """Konfiguration fuer PythonExpert."""

    include_examples: bool = Field(default=True, description="Beispiele einschliessen")
    include_stdlib: bool = Field(default=True, description="Stdlib-Infos einschliessen")
    max_code_lines: int = Field(default=100, ge=1, description="Max Zeilen fuer Code-Generierung")


@register_agent("python_expert")
class PythonExpertAgent(Agent[dict, dict]):
    """
    Python-Experte mit umfassendem Wissen.

    Kann:
    - Python-Konzepte erklaeren
    - Code analysieren und verbessern
    - Code generieren
    - Best Practices empfehlen
    - Fehler debuggen
    """

    agent_type = "python_expert"
    version = "1.0"

    def __init__(self, config: Optional[PythonExpertConfig | dict] = None):
        if config is None:
            config = PythonExpertConfig(name="python_expert")
        elif isinstance(config, dict):
            config = PythonExpertConfig(**config)
        super().__init__(config)
        self.config: PythonExpertConfig = config

    async def execute(self, input_data: dict, context: AgentContext) -> dict:
        """Fuehrt Python-Experten-Aufgaben aus."""
        action = input_data.get("action", "explain")
        query = input_data.get("query", "")
        code = input_data.get("code", "")

        if action == "explain":
            return self._explain_concept(query)
        elif action == "analyze":
            return self._analyze_code(code)
        elif action == "generate":
            return self._generate_code(query)
        elif action == "debug":
            return self._debug_code(code, input_data.get("error", ""))
        elif action == "improve":
            return self._improve_code(code)
        elif action == "search":
            return self._search_knowledge(query)
        elif action == "list_modules":
            return self._list_stdlib_modules()
        elif action == "module_info":
            return self._get_module_info(query)
        else:
            return {"error": f"Unbekannte Aktion: {action}"}

    def _explain_concept(self, query: str) -> dict:
        """Erklaert ein Python-Konzept."""
        query_lower = query.lower().strip()
        results = []

        # Suche in allen Kategorien
        for category, items in PYTHON_KNOWLEDGE.items():
            if category == "design_patterns":
                continue
            for key, value in items.items():
                if query_lower in key.lower() or query_lower in value.lower():
                    results.append({
                        "category": category,
                        "name": key,
                        "description": value,
                    })

        # Design Patterns separat
        if query_lower in ["pattern", "patterns", "design"]:
            for name, code in PYTHON_KNOWLEDGE["design_patterns"].items():
                results.append({
                    "category": "design_patterns",
                    "name": name,
                    "code": code.strip(),
                })

        return {
            "query": query,
            "results": results[:10],
            "total_found": len(results),
        }

    def _analyze_code(self, code: str) -> dict:
        """Analysiert Python-Code."""
        issues = []
        suggestions = []
        info = {}

        try:
            tree = ast.parse(code)

            # Code-Statistiken
            info["lines"] = len(code.splitlines())
            info["functions"] = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            info["classes"] = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            info["imports"] = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])

            # Analyse
            for node in ast.walk(tree):
                # Bare except
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append({
                        "type": "warning",
                        "line": node.lineno,
                        "message": "Bare 'except:' - spezifische Exception verwenden",
                    })

                # Mutable Default Arguments
                if isinstance(node, ast.FunctionDef):
                    for default in node.args.defaults:
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            issues.append({
                                "type": "error",
                                "line": node.lineno,
                                "message": f"Mutable Default-Argument in {node.name}() - None verwenden",
                            })

                    # Fehlender Docstring
                    if not ast.get_docstring(node):
                        suggestions.append({
                            "line": node.lineno,
                            "message": f"Funktion '{node.name}' hat keinen Docstring",
                        })

                # Global Statement
                if isinstance(node, ast.Global):
                    issues.append({
                        "type": "warning",
                        "line": node.lineno,
                        "message": "Global-Statement vermeiden - Klasse oder Parameter verwenden",
                    })

                # print() in Produktionscode
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "print":
                        suggestions.append({
                            "line": node.lineno,
                            "message": "print() durch logging ersetzen",
                        })

            # Best Practice Vorschlaege
            if "import *" in code:
                issues.append({
                    "type": "warning",
                    "message": "'from x import *' vermeiden - explizite Imports verwenden",
                })

        except SyntaxError as e:
            issues.append({
                "type": "error",
                "line": e.lineno,
                "message": f"Syntax-Fehler: {e.msg}",
            })

        return {
            "valid": len([i for i in issues if i["type"] == "error"]) == 0,
            "info": info,
            "issues": issues,
            "suggestions": suggestions,
        }

    def _generate_code(self, query: str) -> dict:
        """Generiert Python-Code basierend auf Beschreibung."""
        query_lower = query.lower()

        # Vordefinierte Code-Templates
        templates = {
            "class": '''class {name}:
    """Beschreibung der Klasse."""

    def __init__(self, {params}):
        {init_body}

    def __repr__(self):
        return f"{name}({self_attrs})"
''',
            "function": '''def {name}({params}) -> {return_type}:
    """
    Beschreibung der Funktion.

    Args:
        {args_doc}

    Returns:
        {return_doc}
    """
    {body}
''',
            "dataclass": '''from dataclasses import dataclass, field
from typing import Optional

@dataclass
class {name}:
    """Beschreibung der Datenklasse."""

    {fields}

    def validate(self) -> bool:
        """Validiert die Daten."""
        return True
''',
            "context_manager": '''from contextlib import contextmanager

@contextmanager
def {name}({params}):
    """Context-Manager fuer {description}."""
    # Setup
    resource = None
    try:
        resource = acquire()
        yield resource
    finally:
        if resource:
            release(resource)
''',
            "decorator": '''import functools
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

def {name}(func: F) -> F:
    """Decorator fuer {description}."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Vor dem Aufruf
        result = func(*args, **kwargs)
        # Nach dem Aufruf
        return result
    return wrapper
''',
            "async_function": '''import asyncio
from typing import Any

async def {name}({params}) -> {return_type}:
    """
    Async-Funktion fuer {description}.

    Args:
        {args_doc}

    Returns:
        {return_doc}
    """
    async with some_resource() as resource:
        result = await resource.operation()
        return result
''',
            "unittest": '''import unittest

class Test{name}(unittest.TestCase):
    """Tests fuer {name}."""

    def setUp(self):
        """Test-Setup."""
        pass

    def tearDown(self):
        """Test-Cleanup."""
        pass

    def test_basic(self):
        """Testet Basis-Funktionalitaet."""
        result = function_under_test()
        self.assertEqual(result, expected)

    def test_edge_case(self):
        """Testet Grenzfaelle."""
        with self.assertRaises(ValueError):
            function_under_test(invalid_input)

if __name__ == "__main__":
    unittest.main()
''',
            "cli": '''import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="{description}")
    parser.add_argument("input", help="Input-Datei")
    parser.add_argument("-o", "--output", default="-", help="Output-Datei")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose Output")

    args = parser.parse_args()

    # Hauptlogik hier
    print(f"Processing {args.input}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
''',
            "api_client": '''import requests
from typing import Any, Optional

class APIClient:
    """HTTP API Client."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def get(self, endpoint: str, params: dict = None) -> dict:
        """GET-Request."""
        response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()

    def post(self, endpoint: str, data: dict) -> dict:
        """POST-Request."""
        response = self.session.post(f"{self.base_url}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
''',
            "singleton": PYTHON_KNOWLEDGE["design_patterns"]["singleton"],
            "factory": PYTHON_KNOWLEDGE["design_patterns"]["factory"],
            "observer": PYTHON_KNOWLEDGE["design_patterns"]["observer"],
        }

        # Finde passendes Template
        for key, template in templates.items():
            if key in query_lower:
                return {
                    "template_type": key,
                    "code": template,
                    "instructions": f"Ersetze die Platzhalter in geschweiften Klammern mit deinen Werten.",
                }

        # Kein Template gefunden - generische Antwort
        return {
            "available_templates": list(templates.keys()),
            "message": "Gib einen Template-Typ an: class, function, dataclass, decorator, async_function, unittest, cli, api_client, singleton, factory, observer",
        }

    def _debug_code(self, code: str, error: str) -> dict:
        """Hilft beim Debuggen von Code."""
        suggestions = []

        error_lower = error.lower()

        # Haeufige Fehler
        if "syntaxerror" in error_lower:
            suggestions.append("Pruefe auf fehlende Klammern, Doppelpunkte oder Einrueckung")
            suggestions.append("Pruefe auf nicht geschlossene Strings oder Klammern")

        if "nameerror" in error_lower:
            suggestions.append("Variable ist nicht definiert - Schreibweise pruefen")
            suggestions.append("Import vergessen?")

        if "typeerror" in error_lower:
            suggestions.append("Falscher Typ fuer Operation")
            suggestions.append("Funktion mit falscher Anzahl Argumente aufgerufen?")
            suggestions.append("None-Wert wo Objekt erwartet?")

        if "indexerror" in error_lower:
            suggestions.append("Index ausserhalb der Liste")
            suggestions.append("len() verwenden um Grenzen zu pruefen")

        if "keyerror" in error_lower:
            suggestions.append("Schluessel existiert nicht im Dictionary")
            suggestions.append(".get(key, default) verwenden")

        if "attributeerror" in error_lower:
            suggestions.append("Objekt hat dieses Attribut nicht")
            suggestions.append("Objekt ist None?")
            suggestions.append("hasattr() zur Pruefung verwenden")

        if "valueerror" in error_lower:
            suggestions.append("Ungueltiger Wert fuer Operation")
            suggestions.append("Input validieren")

        if "importerror" in error_lower or "modulenotfounderror" in error_lower:
            suggestions.append("Modul nicht installiert: pip install <module>")
            suggestions.append("Modulname falsch geschrieben?")

        if "filenotfounderror" in error_lower:
            suggestions.append("Datei existiert nicht")
            suggestions.append("Pfad ueberpruefen")
            suggestions.append("pathlib.Path.exists() zur Pruefung verwenden")

        if "zerodivisionerror" in error_lower:
            suggestions.append("Division durch Null")
            suggestions.append("Divisor vor Division pruefen")

        if "recursionerror" in error_lower:
            suggestions.append("Maximale Rekursionstiefe ueberschritten")
            suggestions.append("Rekursionsbasis pruefen")
            suggestions.append("Iterative Loesung verwenden")

        # Code-Analyse hinzufuegen
        analysis = self._analyze_code(code) if code else {}

        return {
            "error": error,
            "suggestions": suggestions,
            "code_analysis": analysis,
        }

    def _improve_code(self, code: str) -> dict:
        """Schlaegt Verbesserungen fuer Code vor."""
        improvements = []

        # Analyse durchfuehren
        analysis = self._analyze_code(code)

        # String-basierte Checks
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # F-String statt .format()
            if ".format(" in line and "f'" not in line and 'f"' not in line:
                improvements.append({
                    "line": i,
                    "suggestion": "f-String statt .format() verwenden",
                    "example": 'f"Value: {value}" statt "Value: {}".format(value)',
                })

            # % Formatierung
            if "% (" in line or "%s" in line or "%d" in line:
                improvements.append({
                    "line": i,
                    "suggestion": "f-String statt %-Formatierung verwenden",
                })

            # range(len())
            if "range(len(" in line:
                improvements.append({
                    "line": i,
                    "suggestion": "enumerate() statt range(len()) verwenden",
                    "example": "for i, item in enumerate(lst):",
                })

            # == True/False/None
            if "== True" in line or "== False" in line:
                improvements.append({
                    "line": i,
                    "suggestion": "if x: statt if x == True:",
                })
            if "== None" in line or "!= None" in line:
                improvements.append({
                    "line": i,
                    "suggestion": "is None / is not None verwenden",
                })

            # type() statt isinstance()
            if "type(" in line and "==" in line:
                improvements.append({
                    "line": i,
                    "suggestion": "isinstance() statt type() == verwenden",
                })

            # len() == 0
            if "len(" in line and ("== 0" in line or "> 0" in line):
                improvements.append({
                    "line": i,
                    "suggestion": "if not lst: / if lst: statt len() Vergleich",
                })

            # Wildcard import
            if line.strip().startswith("from") and "import *" in line:
                improvements.append({
                    "line": i,
                    "suggestion": "Explizite Imports statt 'import *'",
                })

        return {
            "original_lines": len(lines),
            "analysis": analysis,
            "improvements": improvements,
            "best_practices": PYTHON_KNOWLEDGE["best_practices"],
        }

    def _search_knowledge(self, query: str) -> dict:
        """Durchsucht die Python-Wissensbasis."""
        results = []
        query_lower = query.lower()

        for category, items in PYTHON_KNOWLEDGE.items():
            if isinstance(items, dict):
                for key, value in items.items():
                    if query_lower in key.lower() or (isinstance(value, str) and query_lower in value.lower()):
                        results.append({
                            "category": category,
                            "key": key,
                            "value": value if isinstance(value, str) else "(Code-Beispiel)",
                        })

        return {
            "query": query,
            "results": results[:20],
            "total": len(results),
        }

    def _list_stdlib_modules(self) -> dict:
        """Listet alle Standard-Library Module."""
        return {
            "modules": list(PYTHON_KNOWLEDGE["stdlib_modules"].keys()),
            "total": len(PYTHON_KNOWLEDGE["stdlib_modules"]),
        }

    def _get_module_info(self, module_name: str) -> dict:
        """Gibt Informationen ueber ein Modul."""
        info = PYTHON_KNOWLEDGE["stdlib_modules"].get(module_name)

        if info:
            return {
                "module": module_name,
                "description": info,
                "installed": module_name in sys.modules or self._check_module_available(module_name),
            }

        # Versuche das Modul zu importieren fuer mehr Infos
        try:
            module = __import__(module_name)
            return {
                "module": module_name,
                "description": getattr(module, "__doc__", "Keine Beschreibung"),
                "version": getattr(module, "__version__", "N/A"),
                "file": getattr(module, "__file__", "N/A"),
                "installed": True,
            }
        except ImportError:
            return {
                "module": module_name,
                "error": "Modul nicht gefunden",
                "installed": False,
            }

    def _check_module_available(self, name: str) -> bool:
        """Prueft ob ein Modul verfuegbar ist."""
        try:
            __import__(name)
            return True
        except ImportError:
            return False
