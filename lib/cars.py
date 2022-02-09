#!/usr/bin/env python3

import math
import socket
import time
from enum import Enum

from lib.frame_builders import FrameBuilder


class Direction(Enum):
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    BACKWARD = 4


RESOLUTIONS = {
    0: (96, 96),
    1: (160, 120),
    2: (176, 144),
    3: (240, 176),
    4: (240, 240),
    5: (320, 240),
    6: (400, 296),
    7: (480, 320),
    8: (640, 480),
    9: (800, 600),
    10: (1024, 768),
    11: (1280, 720),
    12: (1280, 1024),
    13: (1600, 1200)
}
DEFAULT_CAMERA_QUALITY = 9
SEND_MOTORS_PERIOD = 0.05

RECV_BUFFER_SIZE = 2048
CONTROL_PORT = 4242
MIN_KEEPALIVE_DELAY = 0.1


def listen_udp(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    return sock


def connect_tcp(ip, port, nodelay=True):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    if nodelay:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


class Camera:
    def __init__(self, connection, quality=DEFAULT_CAMERA_QUALITY):
        self.connection = connection
        self.quality = quality

        self.builder = FrameBuilder()

    def get_frame(self):
        while not self.builder.frames_available():
            self.connection.keep_stream_alive()
            self.builder.take_packet(self.connection.get_packet())

        img = self.builder.ready_frames[-1]
        self.builder.ready_frames = []
        return img

    def get_quality(self):
        return self.quality

    def set_quality(self, quality):
        self.quality = quality

        # Quality is changed and connection object needs to be notified
        # to change the stream messages defining the quality
        self.connection.set_camera_quality(camera_quality=quality)

    def flash_on(self):
        self.connection.send(f"flashlight on\n".encode())

    def flash_off(self):
        self.connection.send(f"flashlight off\n".encode())


class Motors:
    """
    The motor2_multiplier may be used in order to compensate for differences
    in motor speed between the two motors so that the car drives straight
    (i.e., wheels turn with the same speed) when requested so.
    """

    def __init__(self, connection, motor2_multiplier=1.0, verbose=False):
        self.connection = connection
        self.motor2_multiplier = motor2_multiplier
        self.verbose = verbose

    def command(self, power, direction):
        power1 = power
        power2 = power
        if direction == Direction.LEFT:
            power1 *= -1
        elif direction == Direction.RIGHT:
            power2 *= -1
        elif direction == Direction.BACKWARD:
            power1 *= -1
            power2 *= -1
        return self.command_motors(power1, power2)

    def command_motors(self, power_left, power_right):
        power_right = math.copysign(min(abs(power_right * self.motor2_multiplier), 100),
                                    power_right)
        command = f"motors {int(power_left)} {int(power_right)}\n"
        if self.verbose:
            print(command)
        self.connection.send(command.encode())
        return True

    def command_servo(self, position):
        command = f"servo {int(position)}\n"
        if self.verbose:
            print(command)
        self.connection.send(command.encode())
        return True


class Connection:
    def __init__(self, device_ip="192.168.4.1", my_ip="192.168.4.2", recv_port=4545):
        self.my_ip = my_ip
        self.recv_port = recv_port
        self.camera_quality = DEFAULT_CAMERA_QUALITY

        self.sock_control = connect_tcp(device_ip, CONTROL_PORT)
        self.sock_stream = listen_udp(my_ip, recv_port)

        self.last_keepalive = 0
        self.keep_stream_alive()

    def set_camera_quality(self, camera_quality):
        self.camera_quality = camera_quality

    def keep_stream_alive(self):
        now = time.time()
        if now > self.last_keepalive + MIN_KEEPALIVE_DELAY:
            self.last_keepalive = now
            self.sock_control.send(
                f"stream {self.my_ip} {self.recv_port} {self.camera_quality}\n".encode())

    def get_packet(self):
        data, _ = self.sock_stream.recvfrom(RECV_BUFFER_SIZE)
        return data

    def send(self, command):
        self.sock_control.send(command)

    # def __del__(self):
    #     self.sock_control.close()
    #     self.sock_stream.close()
