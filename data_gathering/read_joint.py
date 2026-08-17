import argparse
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()

class JointPositionRecorder:
    def __init__(self, output_location):
        self.output_location = output_location
        self.saved_positions = []
        self.running = True
        self.last_q = None
        
    def save_current_position(self):
        if self.last_q is not None:
            self.saved_positions.append(self.last_q.copy())
            print(f"Saved: {self.last_q.copy()}")
        else:
            print(f"No joint position available")
    
    def save_to_csv(self):
        positions = np.array(self.saved_positions)
        np.savetxt(self.output_location, np.round(positions,8), delimiter=',')
        print(f"saved {positions.shape} positions to {self.output_location}")
    
    def keyboard_listener(self):
        while self.running:
            key = input().strip().upper()
            if key == 'A':
                self.save_current_position()
            elif key == 'S':
                self.save_to_csv()
                self.running = False
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-cfg", type=str, default="joint-position-controller.yml")
    parser.add_argument("--folder", type=Path, default="./positions.csv")
    args = parser.parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )

    save_location = os.path.join(os.path.dirname(__file__), args.folder)
    recorder = JointPositionRecorder(save_location)
    keyboard_thread = threading.Thread(target=recorder.keyboard_listener, daemon=True)
    keyboard_thread.start()

    while recorder.running:
        print(f"last eef: {robot_interface.last_eef_pose}")
        recorder.last_q = robot_interface.last_q
        if recorder.last_q is not None:
            print(f"Current Robot joint: {np.round(robot_interface.last_q, 6)}")
        else:
            print(f"Current Robot joint: {recorder.last_q}")
        time.sleep(1.0)
    
    robot_interface.close()