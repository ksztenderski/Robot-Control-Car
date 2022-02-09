#!/usr/bin/env python3

import numpy as np
from time import sleep

from driving_to_markers import MarkersRobot, angle_to_direction
from lib.cars import Direction
from util import run_robot

basket_position_markers = [7, 9]
basket_pickup_adjust_marker = 4


class BasketsRobot(MarkersRobot):
    def _servo_up(self):
        for pos in range(950, 1501, 10):
            self.motors.command_servo(pos)
            sleep(0.01)

    def _servo_down(self):
        for pos in range(1500, 949, -10):
            self.motors.command_servo(pos)
            sleep(0.01)

    def run(self):
        for marker in self.markers:
            # Drive in distance of the lying markers.
            self.drive_to_marker(marker, stop_distance=500)

            # Position itself close to the 2 lying markers.
            self.find_marker(basket_position_markers, Direction.LEFT)
            self.drive_straight_to_marker(basket_position_markers, stop_distance=300)

            # Turn on the light for clearer colors.
            self.camera_operator.flash_on()
            sleep(0.5)

            while True:
                rvec, tvec = self.get_marker_pose(basket_pickup_adjust_marker)
                dest_marker, left, right = self.camera_operator.detect_basket_color()
                if right - left < 450:
                    dist_from_middle = 520 - (left + right) / 2
                    if np.abs(dist_from_middle) > 50:
                        # Turning.
                        self.run_motors(61, direction=angle_to_direction(dist_from_middle),
                                        sleep_time=0.5)
                    elif tvec is not None and 165 < tvec[2] < 193:
                        # Correct position for pickup.
                        break
                    elif tvec is not None:
                        # Adjusting distance.
                        if tvec[2] < 165:
                            direction = Direction.BACKWARD
                        else:
                            direction = Direction.FORWARD
                        self.run_motors(65, direction=direction, sleep_time=0.5)

            self.camera_operator.flash_off()

            # Pickup basket
            self._servo_up()
            self.run_motors_times(4, 65, direction=Direction.BACKWARD)

            # Drive to the destination marker
            self.drive_to_marker(dest_marker)
            self._servo_down()
            self.run_motors_times(4, 65, direction=Direction.BACKWARD)


if __name__ == '__main__':
    run_robot(BasketsRobot, draw_markers=True, markers=[5, 5])
