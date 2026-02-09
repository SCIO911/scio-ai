#!/usr/bin/env python3
"""
Test fuer den SCIO Universal Super-Agent

Testet alle neuen Module und Faehigkeiten.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def test_nlp():
    """Testet NLP-Modul"""
    print("\n" + "="*60)
    print("TEST: NLP Module")
    print("="*60)

    from scio.nlp import get_nlp

    nlp = get_nlp()

    # Test Text
    text = """
    SCIO is an autonomous AI system that earns money through GPU rental.
    The system uses advanced machine learning to optimize pricing strategies.
    Elon Musk and OpenAI have pioneered similar approaches in the AI industry.
    The current market shows a 15% increase in demand for GPU computing.
    """

    # Vollanalyse
    analysis = nlp.analyze(text)

    print(f"\nLanguage: {analysis.language}")
    print(f"Sentiment: {analysis.sentiment:.2f}")
    print(f"\nKeywords: {analysis.keywords[:5]}")
    print(f"\nEntities found: {len(analysis.entities)}")
    for entity in analysis.entities[:5]:
        print(f"  - {entity.text} ({entity.label})")
    print(f"\nSummary:\n{analysis.summary}")

    print("\n[OK] NLP Module funktioniert!")
    return True


def test_htn_planner():
    """Testet HTN Planner"""
    print("\n" + "="*60)
    print("TEST: HTN Planner")
    print("="*60)

    from scio.planning import get_htn_planner, Task, TaskType

    planner = get_htn_planner()

    # Registrierte Tasks anzeigen
    print(f"\nRegistrierte Tasks: {len(planner.tasks)}")
    for name, task in planner.tasks.items():
        print(f"  - {name} ({task.task_type.value})")

    # Einfachen Plan erstellen
    planner.set_initial_state({"gpu_available": True})
    planner.set_goal({"money_earned": True})

    task = Task(name="earn_money", task_type=TaskType.COMPOUND)
    plan = planner.plan([task])

    print(f"\nPlan valid: {plan.is_valid}")
    print(f"Plan steps: {[t.name for t in plan.tasks]}")
    print(f"Total cost: {plan.total_cost}")

    print("\n[OK] HTN Planner funktioniert!")
    return True


def test_guardrails():
    """Testet Guardrails"""
    print("\n" + "="*60)
    print("TEST: Guardrails System")
    print("="*60)

    from scio.safety import get_guardrails, check_action, ActionType

    guardrails = get_guardrails()

    # Safe Action
    result1 = check_action("Read user preferences", ActionType.READ)
    print(f"\nSafe Action: valid={result1.is_valid}, risk={result1.risk_level.value}")

    # Potentially risky content
    result2 = guardrails.check(
        "Process user data",
        ActionType.DATA_ACCESS,
        content="password=secret123"
    )
    print(f"Risky Content: valid={result2.is_valid}, risk={result2.risk_level.value}")
    if result2.blocked_content:
        print(f"  Blocked: {result2.blocked_content}")

    # Financial check
    result3 = guardrails.action_validator.validate_action(
        ActionType.FINANCIAL,
        {"amount": 75.0}
    )
    print(f"Financial ($75): valid={result3.is_valid}, warnings={result3.warnings}")

    # Risk Summary
    summary = guardrails.get_risk_summary()
    print(f"\nRisk Summary: {summary}")

    print("\n[OK] Guardrails funktioniert!")
    return True


def test_browser_automation():
    """Testet Browser-Automation (ohne echten Browser)"""
    print("\n" + "="*60)
    print("TEST: Browser Automation (Structure Only)")
    print("="*60)

    from scio.web.browser_automation import BrowserAutomation, WebAutomationWorkflow, BrowserType

    # Klassen existieren
    print(f"\nBrowserType values: {[b.value for b in BrowserType]}")

    # Workflow ohne echten Browser
    workflow = WebAutomationWorkflow()
    workflow.navigate("https://example.com")
    workflow.click("#button")
    workflow.fill("#input", "test")
    workflow.screenshot("test")

    print(f"Workflow steps: {len(workflow.steps)}")
    for step in workflow.steps:
        print(f"  - {step['action']}")

    print("\n[OK] Browser Automation Struktur OK!")
    return True


def test_iot_robotics():
    """Testet IoT/Robotics Module"""
    print("\n" + "="*60)
    print("TEST: IoT/Robotics Module")
    print("="*60)

    from scio.hardware.iot_robotics import (
        get_iot_manager, get_robot, get_smart_home,
        IoTDevice, DeviceType, Protocol
    )

    # IoT Manager
    iot = get_iot_manager()

    # Simuliertes Geraet registrieren
    sensor = IoTDevice(
        device_id="temp_sensor_1",
        name="Temperature Sensor",
        device_type=DeviceType.SENSOR,
        protocol=Protocol.MQTT,
        properties={"sensor_type": "temperature", "unit": "celsius", "last_value": 22.5}
    )
    iot.register_device(sensor)

    print(f"\nRegistrierte Geraete: {len(iot.devices)}")

    # Robot Interface
    robot = get_robot()
    robot.set_position(10.0, 20.0, 0.0)
    print(f"Robot Position: {robot.get_position()}")

    # Smart Home
    smart_home = get_smart_home()
    smart_home.add_room("Living Room")

    light = IoTDevice(
        device_id="light_1",
        name="Main Light",
        device_type=DeviceType.LIGHT
    )
    smart_home.add_device_to_room("Living Room", light)

    smart_home.set_scene("evening", {"light_1": {"on": True, "brightness": 50}})
    smart_home.activate_scene("evening")

    status = smart_home.get_room_status("Living Room")
    print(f"Room Status: {status}")

    print("\n[OK] IoT/Robotics funktioniert!")
    return True


def test_social_media():
    """Testet Social Media Integration"""
    print("\n" + "="*60)
    print("TEST: Social Media Integration")
    print("="*60)

    from scio.integrations.social_media import get_social_media_manager, Platform

    social = get_social_media_manager()

    # Verfuegbare Plattformen
    available = social.get_available_platforms()
    print(f"\nVerfuegbare Plattformen: {[p.value for p in available]}")

    # Discord Webhook testen (wenn konfiguriert)
    if social.discord._enabled:
        print("Discord Webhook: Konfiguriert")
    else:
        print("Discord Webhook: Nicht konfiguriert")

    # Telegram testen
    if social.telegram._enabled:
        print("Telegram Bot: Konfiguriert")
    else:
        print("Telegram Bot: Nicht konfiguriert")

    print("\n[OK] Social Media Integration bereit!")
    return True


def test_enterprise():
    """Testet Enterprise Integrations"""
    print("\n" + "="*60)
    print("TEST: Enterprise Integrations")
    print("="*60)

    from scio.integrations.enterprise import get_enterprise_manager, IntegrationType

    enterprise = get_enterprise_manager()

    # Verfuegbare Integrationen
    available = enterprise.get_available_integrations()
    print(f"\nVerfuegbare Integrationen: {[i.value for i in available]}")

    # GitHub testen
    if enterprise.github._enabled:
        print("GitHub: Konfiguriert")
        info = enterprise.github.get_repo_info()
        if info:
            print(f"  Repo: {info.get('full_name', 'N/A')}")
    else:
        print("GitHub: Nicht konfiguriert")

    # Slack testen
    if enterprise.slack._enabled:
        print("Slack: Konfiguriert")
    else:
        print("Slack: Nicht konfiguriert (Webhook verfuegbar: {enterprise.slack.webhook_url != ''})")

    print("\n[OK] Enterprise Integrations bereit!")
    return True


def test_super_agent():
    """Testet den Super-Agent"""
    print("\n" + "="*60)
    print("TEST: Universal Super-Agent")
    print("="*60)

    from scio.super_agent import get_super_agent, CapabilityDomain

    agent = get_super_agent()

    # Summary
    summary = agent.get_summary()
    print(f"\nAgent: {summary['name']}")
    print(f"Version: {summary['version']}")
    print(f"Total Capabilities: {summary['capabilities']['total']}")
    print(f"Domains: {len(summary['capabilities']['domains'])}")

    # Faehigkeiten nach Domain
    print("\nFaehigkeiten pro Domain:")
    for domain, count in summary['capabilities']['by_domain'].items():
        print(f"  {domain}: {count}")

    # State
    state = agent.get_state()
    print(f"\nUptime: {state.uptime_hours:.4f} hours")

    # Text-Analyse testen
    analysis = agent.analyze_text("SCIO is amazing technology from Germany!")
    print(f"\nText Analysis:")
    print(f"  Language: {analysis['language']}")
    print(f"  Sentiment: {analysis['sentiment']:.2f}")
    print(f"  Keywords: {analysis['keywords'][:3]}")

    # Safety Check
    safety = agent.check_safety("Read configuration file")
    print(f"\nSafety Check:")
    print(f"  Valid: {safety['is_valid']}")
    print(f"  Risk: {safety['risk_level']}")

    print("\n[OK] Super-Agent funktioniert!")
    return True


def main():
    """Fuehrt alle Tests aus"""
    print("""
================================================================
     SCIO SUPER-AGENT TEST SUITE
================================================================
     Testing alle neuen Module fuer Universal Digital Super-Agent
================================================================
    """)

    tests = [
        ("NLP", test_nlp),
        ("HTN Planner", test_htn_planner),
        ("Guardrails", test_guardrails),
        ("Browser Automation", test_browser_automation),
        ("IoT/Robotics", test_iot_robotics),
        ("Social Media", test_social_media),
        ("Enterprise", test_enterprise),
        ("Super Agent", test_super_agent),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            results.append((name, False, str(e)))

    # Zusammenfassung
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    passed = 0
    failed = 0

    for name, success, error in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {name}")
        if error:
            print(f"       Error: {error}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n{passed}/{len(results)} tests passed")

    if failed == 0:
        print("""
================================================================
     ALL TESTS PASSED!
     SCIO Universal Super-Agent ist vollstaendig funktionsfaehig!
================================================================
        """)
    else:
        print(f"\n{failed} tests failed - please check errors above")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
