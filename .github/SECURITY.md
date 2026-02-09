# Security Policy

## Unterstützte Versionen

| Version | Unterstützt        |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Sicherheitslücke melden

Wenn du eine Sicherheitslücke in SCIO gefunden hast, melde sie bitte **nicht** öffentlich über GitHub Issues.

### Meldung

1. **E-Mail**: Sende eine E-Mail an security@scio-ai.com
2. **Beschreibung**: Beschreibe die Sicherheitslücke so detailliert wie möglich
3. **Reproduktion**: Füge Schritte zur Reproduktion hinzu
4. **Impact**: Beschreibe den potenziellen Impact

### Was du erwarten kannst

- **Bestätigung**: Innerhalb von 48 Stunden
- **Update**: Innerhalb von 7 Tagen
- **Fix**: Je nach Schweregrad

### Belohnung

Wir schätzen verantwortungsvolle Offenlegung. Abhängig von der Schwere der Sicherheitslücke bieten wir:

- Öffentliche Anerkennung (wenn gewünscht)
- SCIO Premium Zugang

## Best Practices

### Für Benutzer

- Halte SCIO und alle Abhängigkeiten aktuell
- Verwende starke API-Keys
- Speichere keine Credentials im Code
- Verwende Umgebungsvariablen für Secrets

### Für Entwickler

- Folge den OWASP Guidelines
- Führe regelmäßige Security Audits durch
- Halte Abhängigkeiten aktuell (Dependabot)
- Verwende CodeQL für statische Analyse
