#!/usr/bin/env python3

from lib.cars import Direction
from util import RobotBase, run_robot


class SimpleRobot(RobotBase):
    def run(self):
        while True:
            self.connection.keep_stream_alive()
            ids, _, _ = self.camera_operator.detect_markers()

            if ids is not None:
                for direction in [Direction(marker_id) for marker_id in ids if 0 < marker_id < 5]:
                    self.motors.command(80, direction)


if __name__ == '__main__':
    run_robot(SimpleRobot, draw_markers=True)
