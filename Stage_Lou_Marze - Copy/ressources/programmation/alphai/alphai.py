import time
from multiprocessing import connection
import threading
from typing import Union as _U

import numpy as np
from pynput import keyboard


class _Client:
    def __init__(self):
        self.conn = None
        self.robot_connected = False
        self.port = 4741
        self.open_connection()
        self.current_keys = list()
        self.key_changed = threading.Event()
        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release)  # type: threading.Thread
        listener.start()

    def open_connection(self, port=None):
        if port is not None:
            self.port = port
        try:
            self.conn = connection.Client(("localhost", self.port))
            self.robot_connected = self._send("connected", res=True)
            print("Connected to AlphAI, type help(alphai) for more info")
        except ConnectionRefusedError:
            print("No AlphAI instance running yet, you can connect by calling")
            print("alphai.open_api()")

    def __del__(self):
        self.conn.close()

    def _send(self, command, val=None, res=False):
        """
        send a command to the server
        :param command: the command to send, it should match an attribute of
        AlphAIServer
        :param val: the argument of the command, can be None
        :param res: if a response is needed or not, waits for response to
        come if set to True
        :return: None if no response, the response otherwise
        """
        try:
            self.conn.send((command, val))
            if res:
                return self.conn.recv()
        except ConnectionResetError:
            print("No AlphAI instance running")
            if res:
                return False
        except KeyboardInterrupt:
            # stopped during receiving, reset the connection to avoid errors
            self.open_connection()
            raise

    def _on_press(self, key: keyboard.Key):
        k = key.name if hasattr(key, "name") else key.char
        if k not in self.current_keys:
            self.current_keys.append(k)
            self.key_changed.set()

    def _on_release(self, key: keyboard.Key):
        k = key.name if hasattr(key, "name") else key.char
        if k in self.current_keys:
            self.current_keys.remove(k)
            self.key_changed.set()

    def wait_for_key(self):
        # blocking call
        self.key_changed.wait()
        self.key_changed.clear()
        return self.current_keys

    def connect(self, robot_number=True):
        print("attempt connection", robot_number)
        if isinstance(robot_number, bool):
            self.robot_connected = self._send(
                "connect", robot_number, res=True
            )
        elif robot_number == "wifi":
            self.robot_connected = self._send("connect_wifi", res=True)
        else:
            self.robot_connected = self._send(
                "connect_bluetooth", robot_number, res=True)
        return self.robot_connected

    def disconnect(self):
        self._send("connect", False)
        self.robot_connected = self._send("connected", res=True)

    def set_camera(self, resolution=None):
        if resolution is None:
            self._send("set_camera", None)
        elif resolution in [
            f"{2 ** i}x{int(2 ** i * 3 / 4)}" for i in range(1, 7)]:
            self._send("set_camera", resolution)
        else:
            print(f"Unsupported resolution: {resolution}")
            # use KNN to find the closest compatible resolution
            width, height = resolution.split("x")
            ex_res = [(2 ** i, int(2 ** i * 3 / 4)) for i in range(1, 7)]
            try:
                dits = [
                    ((int(width) - i) ** 2 + (int(height) - j) ** 2) ** .5
                    for (i, j) in ex_res
                ]
                s_w, s_h = ex_res[dits.index(min(dits))]
                print(
                    f"did you mean {s_w}x{s_h} ?"
                )
            except ValueError:
                print(f"resolution must be one from: "
                      f"{str([f'{x}x{y}' for x, y in ex_res]).strip('[]')}")

    def set_ir(self, mode):
        if mode in ["mean", "std", "all", "None"] or mode is None:
            self._send("set_ir", mode)
        else:
            print(
                f"Unknown infra-red mode: {mode}, it must be mean, std, "
                f"all or None"
            )

    def set_distance(self, value):
        self._send("set_distance", value)

    def get_distance(self) -> float:
        return self._send("get_distance", res=True) * 100.

    def get_camera(self) -> list:
        image = self._send("get_camera", res=True)
        image = (image * 255).astype(np.uint8).tolist()
        return image

    def get_infrared(self) -> list:
        return self._send("get_ir", res=True).tolist()

    def get_blockade(self) -> bool:
        return self._send("get_blockade", res=True)

    def get_all_sensors(self) -> list:
        return self._send("get_all_sensors", res=True)

    def motor(self, left, right, duration):
        tic = time.perf_counter()
        self._send("motor", (left, right, duration))
        if duration is not None:
            tac = time.perf_counter()
            time.sleep(duration + tic - tac)  # block future calls for the
            # duration of the movement

    def bluetooth_list(self):
        return self._send("bluetooth_list", res=True)


_client = _Client()


def open_api(port=None):
    """
    Open a client and connect it to a running AlphAI instance
    """
    _client.open_connection(port)


def connect(value: _U[str, bool, int] = True):
    """
    Connect to the robot by indicating either 'wifi' for a Wi-Fi connection,
    its number for a Bluetooth connection or True for the default or last used.
    Pass False as argument to disconnect
    @param value: Either True, False, 'wifi' or the robot's number as an
    integer for Bluetooth
    @return: True for a successful connection, False otherwise
    """
    success = _client.connect(value)
    if success:
        print('Connection to robot successful')
    else:
        print('Connection to robot failed')
    return success


def connect_wifi():
    """
    Connect to the robot using Wi-Fi
    @return: True for a successful connection, False otherwise
    """
    return connect('wifi')


def connect_bluetooth(num):
    """
    Connect to the robot using Bluetooth
    @param num: the robot number
    @return: True for a successful connection, False otherwise
    """
    return connect(num)


def disconnect():
    """
    Disconnect from the robot
    """
    _client.disconnect()


def is_connected():
    """
    Get the connection status of the robot
    @return: True if the robot is connected, False otherwise
    """
    return _client.robot_connected


def wait_for_key() -> list:
    """
    blocking call that returns a list of current pressed keys. Is published
    if a key is pressed or released
    :return: a list of the current keys that are being pressed
    """
    return _client.wait_for_key()


def set_camera(resolution=None):
    """
    set the resolution of the camera to the wanted value. The resolution
    must be 2x1 4x3 8x6 16x12 32x24 64x32 or None for no camera to use
    @param resolution: the desired resolution
    """
    _client.set_camera(resolution)


def set_ir(mode):
    """
    set the infra-red trackers to the wanted mode; it must be all std mean
    or None for no ir
    @param mode: the mode to use
    """
    _client.set_ir(mode)


def set_distance(value):
    """
    set the ultrasound sensor
    @param value: True to set it, False to unset it
    """
    _client.set_distance(value)


def get_distance() -> float:
    """
    get the current distance in front of the robot in meters
    @return: the distance in meters
    """
    return _client.get_distance()


def get_camera() -> np.ndarray:
    """
    get the current image of the camera as an array. The way the array is
    organized depends on the chosen resolution
    @return: an array containing the image
    """
    return _client.get_camera()


def get_infra_red() -> list:
    """
    get the current infrared values as an array
    @return: an array containing the infrared values
    """
    return _client.get_infrared()


def get_blockade() -> bool:
    """
    get if the robot is blocked or Not
    @return: True if the robot is not blocked, False if the robot is
    """
    return _client.get_blockade()


def get_all_sensors() -> list:
    """
    get a list of all the sensors in this order:
    - blockade
    - distance
    - infrared
    - camera
    @return: a list of sensors data
    """
    return _client.get_all_sensors()


def motor(left, right, duration=None):
    """
    set the speed of the left and right motor
    @param left: speed of the left motor
    @param right: speed of the right motor
    @param duration: the duration of the movement
    """
    _client.motor(left, right, duration)


def bluetooth_list() -> list:
    """
    get a list of all known robots
    @return: a list of robots
    """
    return _client.bluetooth_list()


if __name__ == '__main__':
    print("Listening to keys, press c to connect/disconnect, b to show known "
          "bluetooth devices, d for distance and e to exit")
    while True:
        current_keys = wait_for_key()
        if 'c' in current_keys:
            if is_connected():
                print("trying to disconnect")
                connect(False)
            else:
                print("trying to connect")
                connected = connect(213)
                print(connected)
        elif 'b' in current_keys:
            print(bluetooth_list())
        elif 'd' in current_keys:
            print(get_distance())
        elif 'e' in current_keys:
            break
        elif 'up' in current_keys:
            if 'left' in current_keys:
                motor(15, 30)
            elif 'right' in current_keys:
                motor(30, 15)
            else:
                motor(30, 30)
        elif 'left' in current_keys:
            motor(0, 30)
        elif 'right' in current_keys:
            motor(30, 0)
        elif 'down' in current_keys:
            motor(-30, -30)
        else:
            motor(0, 0)
