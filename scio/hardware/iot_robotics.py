#!/usr/bin/env python3
"""
SCIO - IoT und Robotics Interface

Ermoeglicht Kommunikation mit:
- IoT-Geraeten (MQTT, HTTP)
- Robotik-Plattformen (ROS-kompatibel)
- Smart Home Systemen
- Industriellen Sensoren
"""

import time
import json
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Thread
from queue import Queue
import socket


class DeviceType(str, Enum):
    """IoT-Geraetetypen"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CAMERA = "camera"
    ROBOT = "robot"
    LIGHT = "light"
    THERMOSTAT = "thermostat"
    SWITCH = "switch"
    MOTOR = "motor"
    DISPLAY = "display"


class DeviceStatus(str, Enum):
    """Geraetestatus"""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    UPDATING = "updating"
    UNKNOWN = "unknown"


class Protocol(str, Enum):
    """Kommunikationsprotokolle"""
    MQTT = "mqtt"
    HTTP = "http"
    WEBSOCKET = "websocket"
    TCP = "tcp"
    UDP = "udp"
    SERIAL = "serial"
    MODBUS = "modbus"
    ROS = "ros"


@dataclass
class SensorReading:
    """Sensormesswert"""
    device_id: str
    sensor_type: str
    value: Any
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    quality: float = 1.0


@dataclass
class IoTDevice:
    """Repraesentiert ein IoT-Geraet"""
    device_id: str
    name: str
    device_type: DeviceType
    protocol: Protocol = Protocol.HTTP
    address: str = ""
    port: int = 0
    status: DeviceStatus = DeviceStatus.UNKNOWN
    properties: Dict[str, Any] = field(default_factory=dict)
    last_seen: float = 0.0
    capabilities: List[str] = field(default_factory=list)

    def is_online(self) -> bool:
        return self.status == DeviceStatus.ONLINE


@dataclass
class RobotState:
    """Zustand eines Roboters"""
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    orientation: Dict[str, float] = field(default_factory=lambda: {"roll": 0, "pitch": 0, "yaw": 0})
    velocity: Dict[str, float] = field(default_factory=lambda: {"linear": 0, "angular": 0})
    joints: Dict[str, float] = field(default_factory=dict)
    battery_level: float = 100.0
    is_moving: bool = False
    current_task: str = ""


class MQTTClient:
    """MQTT Client fuer IoT-Kommunikation"""

    def __init__(self, broker: str = "localhost", port: int = 1883):
        self.broker = broker
        self.port = port
        self._client = None
        self._connected = False
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._message_queue: Queue = Queue()

    def connect(self) -> bool:
        """Verbindet mit MQTT Broker"""
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client()
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.on_disconnect = self._on_disconnect

            self._client.connect(self.broker, self.port, 60)
            self._client.loop_start()

            # Warten auf Verbindung
            timeout = 5.0
            start = time.time()
            while not self._connected and time.time() - start < timeout:
                time.sleep(0.1)

            return self._connected

        except ImportError:
            print("[ERROR] paho-mqtt nicht installiert: pip install paho-mqtt")
            return False
        except Exception as e:
            print(f"[ERROR] MQTT Verbindung fehlgeschlagen: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"[MQTT] Verbunden mit {self.broker}:{self.port}")
            # Subscriptions wiederherstellen
            for topic in self._subscriptions:
                client.subscribe(topic)
        else:
            print(f"[ERROR] MQTT Verbindung fehlgeschlagen: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print("[MQTT] Verbindung getrennt")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = msg.payload.decode()

        self._message_queue.put((topic, payload))

        # Callbacks aufrufen
        if topic in self._subscriptions:
            for callback in self._subscriptions[topic]:
                try:
                    callback(topic, payload)
                except Exception as e:
                    print(f"[ERROR] MQTT Callback Fehler: {e}")

    def subscribe(self, topic: str, callback: Callable = None):
        """Abonniert ein Topic"""
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
            if self._connected:
                self._client.subscribe(topic)

        if callback:
            self._subscriptions[topic].append(callback)

    def publish(self, topic: str, payload: Any, retain: bool = False):
        """Veroeffentlicht eine Nachricht"""
        if not self._connected:
            return False

        if isinstance(payload, dict):
            payload = json.dumps(payload)

        self._client.publish(topic, payload, retain=retain)
        return True

    def get_message(self, timeout: float = 1.0) -> Optional[tuple]:
        """Holt naechste Nachricht aus Queue"""
        try:
            return self._message_queue.get(timeout=timeout)
        except Exception:
            return None

    def disconnect(self):
        """Trennt Verbindung"""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False


class IoTDeviceManager:
    """Verwaltet IoT-Geraete"""

    def __init__(self):
        self.devices: Dict[str, IoTDevice] = {}
        self.sensor_history: Dict[str, List[SensorReading]] = {}
        self._mqtt: Optional[MQTTClient] = None
        self._callbacks: Dict[str, List[Callable]] = {}

    def register_device(self, device: IoTDevice):
        """Registriert ein Geraet"""
        self.devices[device.device_id] = device
        self.sensor_history[device.device_id] = []
        print(f"[IOT] Geraet registriert: {device.name} ({device.device_id})")

    def get_device(self, device_id: str) -> Optional[IoTDevice]:
        """Gibt Geraet zurueck"""
        return self.devices.get(device_id)

    def list_devices(self, device_type: DeviceType = None) -> List[IoTDevice]:
        """Listet Geraete"""
        devices = list(self.devices.values())
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
        return devices

    def connect_mqtt(self, broker: str = "localhost", port: int = 1883) -> bool:
        """Verbindet mit MQTT Broker"""
        self._mqtt = MQTTClient(broker, port)
        return self._mqtt.connect()

    def send_command(self, device_id: str, command: str, params: Dict = None) -> bool:
        """Sendet Befehl an Geraet"""
        device = self.devices.get(device_id)
        if not device:
            return False

        payload = {
            "command": command,
            "params": params or {},
            "timestamp": time.time()
        }

        if device.protocol == Protocol.MQTT and self._mqtt:
            topic = f"scio/devices/{device_id}/commands"
            return self._mqtt.publish(topic, payload)

        elif device.protocol == Protocol.HTTP:
            return self._send_http_command(device, payload)

        return False

    def _send_http_command(self, device: IoTDevice, payload: Dict) -> bool:
        """Sendet HTTP-Befehl"""
        try:
            import requests
            url = f"http://{device.address}:{device.port}/command"
            response = requests.post(url, json=payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"[ERROR] HTTP Befehl fehlgeschlagen: {e}")
            return False

    def read_sensor(self, device_id: str) -> Optional[SensorReading]:
        """Liest Sensorwert"""
        device = self.devices.get(device_id)
        if not device or device.device_type != DeviceType.SENSOR:
            return None

        # Simulierte Lesung (in Produktion: echte Kommunikation)
        reading = SensorReading(
            device_id=device_id,
            sensor_type=device.properties.get("sensor_type", "unknown"),
            value=device.properties.get("last_value", 0),
            unit=device.properties.get("unit", "")
        )

        self.sensor_history[device_id].append(reading)
        return reading

    def get_sensor_history(self, device_id: str, limit: int = 100) -> List[SensorReading]:
        """Gibt Sensor-Historie zurueck"""
        history = self.sensor_history.get(device_id, [])
        return history[-limit:]

    def on_device_event(self, device_id: str, callback: Callable):
        """Registriert Event-Callback"""
        if device_id not in self._callbacks:
            self._callbacks[device_id] = []
        self._callbacks[device_id].append(callback)


class RobotInterface:
    """Interface fuer Roboter-Steuerung (ROS-kompatibel)"""

    def __init__(self, robot_id: str = "scio_robot"):
        self.robot_id = robot_id
        self.state = RobotState()
        self._ros_available = False
        self._publisher = None
        self._subscriber = None

        # ROS initialisieren wenn verfuegbar
        self._try_init_ros()

    def _try_init_ros(self):
        """Versucht ROS zu initialisieren"""
        try:
            import rospy
            from geometry_msgs.msg import Twist, Pose
            from sensor_msgs.msg import JointState

            if not rospy.core.is_initialized():
                rospy.init_node(f'scio_{self.robot_id}', anonymous=True)

            self._ros_available = True
            print(f"[ROS] Initialisiert fuer Robot: {self.robot_id}")

        except ImportError:
            print("[INFO] ROS nicht verfuegbar - Simulations-Modus")
            self._ros_available = False

    def move(self, linear: float = 0.0, angular: float = 0.0) -> bool:
        """Bewegt den Roboter"""
        if self._ros_available:
            return self._ros_move(linear, angular)

        # Simulations-Modus
        self.state.velocity = {"linear": linear, "angular": angular}
        self.state.is_moving = linear != 0 or angular != 0
        return True

    def _ros_move(self, linear: float, angular: float) -> bool:
        """Bewegt Roboter ueber ROS"""
        try:
            import rospy
            from geometry_msgs.msg import Twist

            pub = rospy.Publisher(f'/{self.robot_id}/cmd_vel', Twist, queue_size=10)

            twist = Twist()
            twist.linear.x = linear
            twist.angular.z = angular

            pub.publish(twist)
            return True

        except Exception as e:
            print(f"[ERROR] ROS Move fehlgeschlagen: {e}")
            return False

    def stop(self) -> bool:
        """Stoppt den Roboter"""
        return self.move(0.0, 0.0)

    def set_position(self, x: float, y: float, z: float = 0.0):
        """Setzt Zielposition"""
        self.state.position = {"x": x, "y": y, "z": z}

    def get_position(self) -> Dict[str, float]:
        """Gibt aktuelle Position zurueck"""
        return self.state.position.copy()

    def set_joint(self, joint_name: str, angle: float) -> bool:
        """Setzt Gelenkwinkel"""
        self.state.joints[joint_name] = angle

        if self._ros_available:
            return self._ros_set_joint(joint_name, angle)
        return True

    def _ros_set_joint(self, joint_name: str, angle: float) -> bool:
        """Setzt Gelenk ueber ROS"""
        try:
            import rospy
            from std_msgs.msg import Float64

            topic = f'/{self.robot_id}/joint_{joint_name}_controller/command'
            pub = rospy.Publisher(topic, Float64, queue_size=10)
            pub.publish(angle)
            return True
        except Exception as e:
            print(f"[ERROR] ROS Joint fehlgeschlagen: {e}")
            return False

    def get_state(self) -> RobotState:
        """Gibt aktuellen Zustand zurueck"""
        return self.state

    def execute_trajectory(self, waypoints: List[Dict[str, float]], speed: float = 1.0) -> bool:
        """Fuehrt Trajektorie aus"""
        for wp in waypoints:
            self.set_position(
                wp.get("x", 0),
                wp.get("y", 0),
                wp.get("z", 0)
            )
            # Simulierte Bewegungszeit
            time.sleep(1.0 / speed)

        return True

    def gripper_open(self) -> bool:
        """Oeffnet Greifer"""
        return self.set_joint("gripper", 1.0)

    def gripper_close(self) -> bool:
        """Schliesst Greifer"""
        return self.set_joint("gripper", 0.0)


class SmartHomeHub:
    """Smart Home Integration"""

    def __init__(self):
        self.devices: Dict[str, IoTDevice] = {}
        self.rooms: Dict[str, List[str]] = {}
        self.scenes: Dict[str, Dict] = {}

    def add_room(self, room_name: str):
        """Fuegt Raum hinzu"""
        if room_name not in self.rooms:
            self.rooms[room_name] = []

    def add_device_to_room(self, room_name: str, device: IoTDevice):
        """Fuegt Geraet zu Raum hinzu"""
        self.add_room(room_name)
        self.rooms[room_name].append(device.device_id)
        self.devices[device.device_id] = device

    def set_scene(self, scene_name: str, settings: Dict[str, Any]):
        """Definiert eine Szene"""
        self.scenes[scene_name] = settings

    def activate_scene(self, scene_name: str) -> bool:
        """Aktiviert eine Szene"""
        if scene_name not in self.scenes:
            return False

        settings = self.scenes[scene_name]
        for device_id, state in settings.items():
            if device_id in self.devices:
                self.devices[device_id].properties.update(state)

        print(f"[SMART HOME] Szene aktiviert: {scene_name}")
        return True

    def control_light(self, device_id: str, on: bool = True, brightness: int = 100, color: str = None):
        """Steuert Licht"""
        device = self.devices.get(device_id)
        if not device or device.device_type != DeviceType.LIGHT:
            return False

        device.properties.update({
            "on": on,
            "brightness": brightness,
            "color": color
        })
        return True

    def set_thermostat(self, device_id: str, temperature: float, mode: str = "auto"):
        """Setzt Thermostat"""
        device = self.devices.get(device_id)
        if not device or device.device_type != DeviceType.THERMOSTAT:
            return False

        device.properties.update({
            "target_temperature": temperature,
            "mode": mode
        })
        return True

    def get_room_status(self, room_name: str) -> Dict[str, Any]:
        """Gibt Raumstatus zurueck"""
        if room_name not in self.rooms:
            return {}

        status = {"room": room_name, "devices": {}}

        for device_id in self.rooms[room_name]:
            if device_id in self.devices:
                device = self.devices[device_id]
                status["devices"][device_id] = {
                    "name": device.name,
                    "type": device.device_type.value,
                    "status": device.status.value,
                    "properties": device.properties
                }

        return status


# Singletons
_device_manager: Optional[IoTDeviceManager] = None
_robot: Optional[RobotInterface] = None
_smart_home: Optional[SmartHomeHub] = None


def get_iot_manager() -> IoTDeviceManager:
    """Gibt IoT Device Manager zurueck"""
    global _device_manager
    if _device_manager is None:
        _device_manager = IoTDeviceManager()
    return _device_manager


def get_robot() -> RobotInterface:
    """Gibt Robot Interface zurueck"""
    global _robot
    if _robot is None:
        _robot = RobotInterface()
    return _robot


def get_smart_home() -> SmartHomeHub:
    """Gibt Smart Home Hub zurueck"""
    global _smart_home
    if _smart_home is None:
        _smart_home = SmartHomeHub()
    return _smart_home
