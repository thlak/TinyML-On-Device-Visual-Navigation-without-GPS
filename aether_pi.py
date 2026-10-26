"""MQTT dashboard client for Raspberry Pi video and inference results.

`Aether.send_frame()` publishes JPEG video and `Aether.send()` publishes JSON
results. The standalone `main()` streams video only; inference runs in `infer.py`.
"""
import io
import json
import os
import time

import paho.mqtt.client as mqtt

HOST = "tinyml.mameed.com"
TOPIC = "group30/pi1"
USER = "pi"
FPS = 20

class Aether:
    """Connection to the MQTT broker on our server."""

    def __init__(self):
        password = os.environ.get("AETHER_MQTT_PASSWORD")
        if not password:
            raise RuntimeError("Set AETHER_MQTT_PASSWORD before connecting to the dashboard.")

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="websockets")
        self.client.ws_set_options(path="/mqtt")
        self.client.tls_set()
        self.client.username_pw_set(USER, password)

        # If the Pi suddenly disconnects, the server tells the dashboard "offline" for us
        self.client.will_set(TOPIC + "/status", "offline", retain=True)
        self.client.on_connect = self.on_connect

        self.client.connect(HOST, 443, keepalive=15)  # "offline" within ~20 s if the Pi dies
        self.client.loop_start()  # network runs in the background, reconnects by itself

    def on_connect(self, client, userdata, flags, reason, properties):
        print("Connected to the server:", reason)
        client.publish(TOPIC + "/status", "online", retain=True)

    def send(self, data):
        """data is a dict, e.g. {"command": "left", "latency": 38}. Format: see the dashboard's pi.service.js"""
        self.client.publish(TOPIC + "/data", json.dumps(data))

    def send_frame(self, jpeg_bytes):
        self.client.publish(TOPIC + "/frame", jpeg_bytes)

    def close(self):
        self.client.publish(TOPIC + "/status", "offline", retain=True).wait_for_publish()
        self.client.disconnect()


def main():
    from picamera2 import Picamera2

    camera = Picamera2()
    camera.configure(camera.create_video_configuration(main={"size": (320, 240)}))
    camera.start()

    aether = Aether()
    try:
        while True:
            started = time.time()

            # 1. Video for the dashboard
            jpeg = io.BytesIO()
            camera.capture_file(jpeg, format="jpeg")
            aether.send_frame(jpeg.getvalue())

            # 2. TODO: run the model here, e.g.
            # frame = camera.capture_array()
            # result = model(frame)
            # aether.send({"command": ..., "target": ..., "latency": ...})

            # Wait so that we send about FPS frames per second
            time.sleep(max(0, 1 / FPS - (time.time() - started)))
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        aether.close()


if __name__ == "__main__":
    main()
