#!/usr/bin/env python3

import numpy as np
from time import sleep

from lib.cars import Direction
from util import RobotBase, run_robot

# Difference in motors power to keep the direction relatively straight.
straight_seq = [-2, -1]

# Distance on Z marker axis towards which robot drives.
cross_distance = 700

# Angle considered to be straight ~ [-2.86°, 2.86°].
threshold_angle = 0.05


def angle_to_direction(angle):
    if angle < 0:
        return Direction.RIGHT
    else:
        return Direction.LEFT


def _opposite_direction(direction):
    return Direction(3 - direction.value)


def _tvec_to_distance(tvec):
    return np.sqrt(tvec[1] ** 2 + tvec[2] ** 2)


class MarkersRobot(RobotBase):
    def __init__(self, connection, markers=None, **kwargs):
        super().__init__(connection, **kwargs)

        # Markers towards which robot will drive.
        self.markers = [] if markers is None else markers

    def run_motors(self, power_left, power_right=None, direction=None, sleep_time=0.3):
        self.connection.keep_stream_alive()
        if power_right is None:
            self.motors.command(power_left, direction)
        else:
            self.motors.command_motors(power_left, power_right)
        sleep(sleep_time)

    def run_motors_times(self, n_steps, power_left, power_right=None, direction=None):
        for _ in range(n_steps):
            self.run_motors(power_left, power_right, direction)

    def get_marker_pose(self, marker_id):
        """
        Returns position of a marker. If many marker ids specified returns average position of the
        detected subset of wanted markers.
        """

        if not isinstance(marker_id, list):
            marker_id = [marker_id]

        ids, rvecs, tvecs = self.camera_operator.detect_markers()
        if ids is not None and any([i in ids for i in marker_id]):
            idx = np.where(np.any(np.array([ids == i for i in marker_id]), axis=0))[0]
            return rvecs[idx].mean(axis=0), tvecs[idx].mean(axis=0)

        return None, None

    def find_marker(self, marker_id, turn_direction, lookup=True, threshold=threshold_angle):
        """ Position the robot towards the marker."""

        tan = 1
        while tan > threshold:
            _, tvec = self.get_marker_pose(marker_id)
            if tvec is None and lookup:
                # No visible marker, turning faster.
                self.run_motors(67, direction=turn_direction, sleep_time=1)
            elif tvec is not None:
                # Visible marker, turning slower.
                tan = np.abs(tvec[0] / tvec[2])
                if tan > threshold:
                    self.run_motors(62, direction=angle_to_direction(-tvec[0]), sleep_time=1)

    def drive_straight_to_marker(self, marker_id, stop_distance=300,
                                 turning_threshold=threshold_angle):
        i = 0
        while True:
            _, tvec = self.get_marker_pose(marker_id)
            if tvec is not None:
                if tvec[0] / tvec[2] > turning_threshold:
                    # Reposition itself towards the marker.
                    self.find_marker(marker_id, Direction.LEFT)

                distance = _tvec_to_distance(tvec)
                if distance < stop_distance:
                    break

                # Power adjusted to the distance from the marker.
                power = min(73 + (distance - stop_distance) // 100, 80)
                self.run_motors(power, power + straight_seq[i % 2])
                i += 1

    def drive_to_marker(self, marker_id, stop_distance=300):
        direction = Direction.RIGHT

        while True:
            # Turn towards the marker
            self.find_marker(marker_id, _opposite_direction(direction))
            rvec, tvec = None, None

            while rvec is None:
                rvec, tvec = self.get_marker_pose(marker_id)
            angle = rvec[1]

            # Calculate approximation of the road.
            distance_to_marker = _tvec_to_distance(tvec)
            distance_to_axis = np.abs(np.sin(angle) * distance_to_marker)

            if distance_to_axis < 50:
                break

            # Distance to travel to the point <cross_distance> from marker on the Z axis.
            travel_to_axis = np.sqrt(cross_distance ** 2 + distance_to_marker ** 2 -
                                     2 * cross_distance * distance_to_marker * np.cos(angle))
            turn_angle = np.abs(np.arcsin((cross_distance * np.sin(angle)) / travel_to_axis))
            turn_steps = int(turn_angle * 5.73)  # 1/5.73 rad ~ 10°
            move_steps = int(np.ceil(travel_to_axis / 50))
            direction = angle_to_direction(angle)

            # Turn towards the axis.
            self.run_motors_times(turn_steps, 65, direction=direction)

            # Drive towards the axis.
            for i in range(move_steps):
                self.run_motors(74, 74 + straight_seq[i % len(straight_seq)])

        # Moving along the Z axis.
        self.find_marker(marker_id, _opposite_direction(direction))
        self.drive_straight_to_marker(marker_id, stop_distance)
        self.camera_operator.flash()
        sleep(1)

    def run(self):
        for marker_id in self.markers:
            self.drive_to_marker(marker_id)


if __name__ == '__main__':
    run_robot(MarkersRobot, draw_markers=False, draw_axis=True, markers=[0, 1, 2, 3])
