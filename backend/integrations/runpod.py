#!/usr/bin/env python3
"""
SCIO - RunPod Integration
Serverless GPU Platform
"""

import json
import time
import threading
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

import requests

from backend.config import Config


@dataclass
class RunPodEndpoint:
    """RunPod Serverless Endpoint"""
    endpoint_id: str
    name: str
    gpu_type: str
    workers_min: int
    workers_max: int
    idle_timeout: int
    status: str
    requests_total: int
    requests_completed: int
    requests_failed: int


@dataclass
class RunPodPod:
    """RunPod GPU Pod"""
    pod_id: str
    name: str
    gpu_type: str
    gpu_count: int
    status: str
    cost_per_hour: float
    uptime_seconds: int
    ssh_command: str


class RunPodIntegration:
    """
    RunPod Integration

    Features:
    - Serverless Endpoints
    - Pod Management
    - Usage Tracking
    - Auto-Scaling
    """

    API_BASE = "https://api.runpod.io/graphql"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.RUNPOD_API_KEY
        self._enabled = Config.RUNPOD_ENABLED and bool(self.api_key)
        self._endpoints: Dict[str, RunPodEndpoint] = {}
        self._pods: Dict[str, RunPodPod] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._total_spent = 0.0
        self._callbacks: List = []

        if not self._enabled:
            print("[WARN]  RunPod Integration deaktiviert")

    def _graphql(self, query: str, variables: dict = None) -> dict:
        """Macht GraphQL-Request"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        try:
            resp = requests.post(
                self.API_BASE,
                headers=headers,
                json={'query': query, 'variables': variables or {}},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

            if 'errors' in result:
                raise Exception(result['errors'][0]['message'])

            return result.get('data', {})

        except requests.RequestException as e:
            print(f"[ERROR] RunPod API Fehler: {e}")
            raise

    def add_callback(self, callback):
        """Registriert Callback für Events"""
        self._callbacks.append(callback)

    def _notify(self, event: str, data: dict = None):
        """Benachrichtigt Callbacks"""
        for callback in self._callbacks:
            try:
                callback(event, data or {})
            except Exception:
                pass  # Callback errors should not affect operation

    def get_myself(self) -> dict:
        """Gibt Account-Info zurück"""
        if not self._enabled:
            return {'enabled': False}

        query = """
        query {
            myself {
                id
                email
                currentSpendPerHr
                machineQuota
                referralEarned
                creditBalance
                notifyPodsStale
                notifyPodsGeneral
            }
        }
        """

        try:
            result = self._graphql(query)
            return result.get('myself', {})
        except Exception as e:
            print(f"[ERROR] Account-Info abrufen fehlgeschlagen: {e}")
            return {}

    def get_pods(self) -> List[RunPodPod]:
        """Gibt alle Pods zurück"""
        if not self._enabled:
            return []

        query = """
        query {
            myself {
                pods {
                    id
                    name
                    runtime {
                        uptimeInSeconds
                        ports {
                            ip
                            privatePort
                            publicPort
                            type
                        }
                    }
                    machine {
                        gpuDisplayName
                    }
                    gpuCount
                    costPerHr
                    desiredStatus
                }
            }
        }
        """

        try:
            result = self._graphql(query)
            pods_data = result.get('myself', {}).get('pods', [])
            pods = []

            for p in pods_data:
                runtime = p.get('runtime') or {}
                machine = p.get('machine') or {}

                # Build SSH command
                ssh_cmd = ""
                for port in runtime.get('ports', []):
                    if port.get('privatePort') == 22:
                        ssh_cmd = f"ssh root@{port['ip']} -p {port['publicPort']}"
                        break

                pod = RunPodPod(
                    pod_id=p['id'],
                    name=p.get('name', 'Unnamed'),
                    gpu_type=machine.get('gpuDisplayName', 'Unknown'),
                    gpu_count=p.get('gpuCount', 1),
                    status=p.get('desiredStatus', 'unknown'),
                    cost_per_hour=p.get('costPerHr', 0),
                    uptime_seconds=runtime.get('uptimeInSeconds', 0),
                    ssh_command=ssh_cmd,
                )
                pods.append(pod)
                self._pods[pod.pod_id] = pod

            return pods

        except Exception as e:
            print(f"[ERROR] Pods abrufen fehlgeschlagen: {e}")
            return []

    def get_endpoints(self) -> List[RunPodEndpoint]:
        """Gibt Serverless Endpoints zurück"""
        if not self._enabled:
            return []

        query = """
        query {
            myself {
                serverlessDiscount
                endpoints {
                    id
                    name
                    gpuIds
                    workersMin
                    workersMax
                    idleTimeout
                }
            }
        }
        """

        try:
            result = self._graphql(query)
            endpoints_data = result.get('myself', {}).get('endpoints', [])
            endpoints = []

            for e in endpoints_data:
                endpoint = RunPodEndpoint(
                    endpoint_id=e['id'],
                    name=e.get('name', 'Unnamed'),
                    gpu_type=e.get('gpuIds', ['Unknown'])[0] if e.get('gpuIds') else 'Unknown',
                    workers_min=e.get('workersMin', 0),
                    workers_max=e.get('workersMax', 1),
                    idle_timeout=e.get('idleTimeout', 5),
                    status='active',
                    requests_total=0,
                    requests_completed=0,
                    requests_failed=0,
                )
                endpoints.append(endpoint)
                self._endpoints[endpoint.endpoint_id] = endpoint

            return endpoints

        except Exception as e:
            print(f"[ERROR] Endpoints abrufen fehlgeschlagen: {e}")
            return []

    def create_pod(
        self,
        name: str,
        gpu_type: str = "NVIDIA RTX A4000",
        gpu_count: int = 1,
        image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        volume_size: int = 20,
    ) -> Optional[str]:
        """Erstellt neuen Pod"""
        if not self._enabled:
            return None

        query = """
        mutation($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
            }
        }
        """

        variables = {
            'input': {
                'name': name,
                'gpuTypeId': gpu_type,
                'gpuCount': gpu_count,
                'imageName': image,
                'volumeInGb': volume_size,
                'containerDiskInGb': 20,
                'dockerArgs': '',
                'volumeMountPath': '/workspace',
            }
        }

        try:
            result = self._graphql(query, variables)
            pod = result.get('podFindAndDeployOnDemand', {})
            pod_id = pod.get('id')

            if pod_id:
                print(f"[OK] Pod erstellt: {pod_id}")
                return pod_id
            return None

        except Exception as e:
            print(f"[ERROR] Pod erstellen fehlgeschlagen: {e}")
            return None

    def stop_pod(self, pod_id: str) -> bool:
        """Stoppt einen Pod"""
        if not self._enabled:
            return False

        query = """
        mutation($input: PodStopInput!) {
            podStop(input: $input) {
                id
                desiredStatus
            }
        }
        """

        try:
            self._graphql(query, {'input': {'podId': pod_id}})
            print(f"[STOP] Pod gestoppt: {pod_id}")
            return True

        except Exception as e:
            print(f"[ERROR] Pod stoppen fehlgeschlagen: {e}")
            return False

    def resume_pod(self, pod_id: str) -> bool:
        """Startet einen gestoppten Pod"""
        if not self._enabled:
            return False

        query = """
        mutation($input: PodResumeInput!) {
            podResume(input: $input) {
                id
                desiredStatus
            }
        }
        """

        try:
            self._graphql(query, {'input': {'podId': pod_id}})
            print(f"[RUN]  Pod gestartet: {pod_id}")
            return True

        except Exception as e:
            print(f"[ERROR] Pod starten fehlgeschlagen: {e}")
            return False

    def terminate_pod(self, pod_id: str) -> bool:
        """Löscht einen Pod permanent"""
        if not self._enabled:
            return False

        query = """
        mutation($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """

        try:
            self._graphql(query, {'input': {'podId': pod_id}})
            print(f"🗑️  Pod gelöscht: {pod_id}")
            return True

        except Exception as e:
            print(f"[ERROR] Pod löschen fehlgeschlagen: {e}")
            return False

    def call_endpoint(
        self,
        endpoint_id: str,
        input_data: dict,
        timeout: int = 60,
    ) -> Optional[dict]:
        """Ruft Serverless Endpoint auf"""
        if not self._enabled:
            return None

        url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json={'input': input_data},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            print(f"[ERROR] Endpoint-Aufruf fehlgeschlagen: {e}")
            return None

    def get_spending(self) -> dict:
        """Gibt Ausgaben-Übersicht zurück"""
        if not self._enabled:
            return {'enabled': False}

        try:
            account = self.get_myself()
            pods = self.get_pods()

            current_hourly = sum(p.cost_per_hour for p in pods if p.status == 'RUNNING')
            daily_estimate = current_hourly * 24
            monthly_estimate = daily_estimate * 30

            return {
                'enabled': True,
                'credit_balance': account.get('creditBalance', 0),
                'current_spend_per_hour': account.get('currentSpendPerHr', 0),
                'current_hourly_usd': round(current_hourly, 2),
                'daily_estimate_usd': round(daily_estimate, 2),
                'monthly_estimate_usd': round(monthly_estimate, 2),
                'active_pods': len([p for p in pods if p.status == 'RUNNING']),
                'total_pods': len(pods),
            }

        except Exception as e:
            print(f"[ERROR] Spending abrufen fehlgeschlagen: {e}")
            return {'enabled': True, 'error': str(e)}

    def _monitor_loop(self):
        """Monitoring Loop"""
        while self._running:
            try:
                # Refresh pods
                pods = self.get_pods()

                # Check for running pods
                running_pods = [p for p in pods if p.status == 'RUNNING']
                for pod in running_pods:
                    self._notify('pod_running', {
                        'pod_id': pod.pod_id,
                        'name': pod.name,
                        'cost_per_hour': pod.cost_per_hour,
                    })

                # Update spending
                spending = self.get_spending()
                self._total_spent = spending.get('current_spend_per_hour', 0) * 24 * 30

            except Exception as e:
                print(f"[WARN]  RunPod Monitor Fehler: {e}")

            time.sleep(300)  # 5 Minuten

    def start_monitor(self):
        """Startet Background-Monitor"""
        if not self._enabled or self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("[OK] RunPod Monitor gestartet")

    def stop_monitor(self):
        """Stoppt Monitor"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        print("[STOP] RunPod Monitor gestoppt")

    def get_status(self) -> dict:
        """Gibt Integration-Status zurück"""
        return {
            'enabled': self._enabled,
            'running': self._running,
            'pods': len(self._pods),
            'endpoints': len(self._endpoints),
            'active_pods': len([p for p in self._pods.values() if p.status == 'RUNNING']),
        }


# Singleton Instance
_runpod_instance: Optional[RunPodIntegration] = None


def get_runpod() -> RunPodIntegration:
    """Gibt Singleton-Instanz zurück"""
    global _runpod_instance
    if _runpod_instance is None:
        _runpod_instance = RunPodIntegration()
    return _runpod_instance
