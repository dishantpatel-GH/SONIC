#!/usr/bin/env python3
"""Standalone Inspire DFX hand test — directly controls fingers via DDS.

Replicates what the hand_example binary does: sends [0,1] normalized commands
to the Inspire hands over the rt/inspire/cmd DDS topic.

Hardware motor layout (from Unitree docs):
  DDS Index | Joint (Right Hand) | Joint (Left Hand)
  ----------|--------------------|-------------------
  0  / 6    | pinky              | pinky
  1  / 7    | ring               | ring
  2  / 8    | middle             | middle
  3  / 9    | index              | index
  4  / 10   | thumb bend (pitch) | thumb bend (pitch)
  5  / 11   | thumb rotation     | thumb rotation

Command values: 0.0 = fully closed, 1.0 = fully open

Usage:
  python test_inspire_hands.py              # Test right hand (default)
  python test_inspire_hands.py --left       # Test left hand
  python test_inspire_hands.py --both       # Test both hands
  python test_inspire_hands.py --finger 3   # Test only index finger (DDS idx 3)
"""

import argparse
import time
import threading
import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

TOPIC_CMD = "rt/inspire/cmd"
TOPIC_STATE = "rt/inspire/state"
NUM_MOTORS = 12  # 0-5 right, 6-11 left
MOTORS_PER_HAND = 6

# Joint names per DDS index (within a hand)
JOINT_NAMES = ["pinky", "ring", "middle", "index", "thumb_pitch", "thumb_yaw"]

# Latest state from subscriber
latest_state = None
state_lock = threading.Lock()


def state_callback(msg):
    global latest_state
    with state_lock:
        latest_state = msg


def build_cmd(right_q, left_q):
    """Build a MotorCmds_ message from right[6] and left[6] arrays of [0,1] values."""
    msg = MotorCmds_()
    msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(NUM_MOTORS)]
    for i in range(MOTORS_PER_HAND):
        msg.cmds[i].mode = 1
        msg.cmds[i].q = float(right_q[i])
    for i in range(MOTORS_PER_HAND):
        msg.cmds[i + 6].mode = 1
        msg.cmds[i + 6].q = float(left_q[i])
    return msg


def print_state():
    """Print current hand state from DDS feedback."""
    with state_lock:
        s = latest_state
    if s is None or len(s.states) < NUM_MOTORS:
        print("  [state] No state data yet")
        return
    right_q = [s.states[i].q for i in range(6)]
    left_q = [s.states[i + 6].q for i in range(6)]
    r_str = " ".join(f"{v:.3f}" for v in right_q)
    l_str = " ".join(f"{v:.3f}" for v in left_q)
    print(f"  [state] R: [{r_str}]  L: [{l_str}]")
    print(f"          R: [{', '.join(f'{JOINT_NAMES[i]}={right_q[i]:.3f}' for i in range(6))}]")


def send_and_hold(pub, right_q, left_q, duration=1.0, hz=100):
    """Send command repeatedly for `duration` seconds at `hz` rate."""
    msg = build_cmd(right_q, left_q)
    steps = int(duration * hz)
    for _ in range(steps):
        pub.Write(msg)
        time.sleep(1.0 / hz)


def ramp_to(pub, right_target, left_target, right_current, left_current, duration=1.0, hz=100):
    """Smoothly ramp from current to target over duration."""
    steps = int(duration * hz)
    for step in range(steps + 1):
        t = step / max(steps, 1)
        right_q = [right_current[i] + t * (right_target[i] - right_current[i]) for i in range(6)]
        left_q = [left_current[i] + t * (left_target[i] - left_current[i]) for i in range(6)]
        msg = build_cmd(right_q, left_q)
        pub.Write(msg)
        time.sleep(1.0 / hz)
    return right_target[:], left_target[:]


def main():
    parser = argparse.ArgumentParser(description="Test Inspire DFX hands via DDS")
    parser.add_argument("--left", action="store_true", help="Test left hand")
    parser.add_argument("--both", action="store_true", help="Test both hands")
    parser.add_argument("--finger", type=int, default=-1, help="Test single DDS motor index (0-5)")
    parser.add_argument("--cycles", type=int, default=3, help="Number of open/close cycles")
    parser.add_argument("--speed", type=float, default=1.0, help="Ramp duration in seconds")
    args = parser.parse_args()

    test_right = not args.left or args.both
    test_left = args.left or args.both

    print("=" * 60)
    print("Inspire DFX Hand Test")
    print("=" * 60)
    print(f"Testing: {'RIGHT' if test_right else ''} {'LEFT' if test_left else ''}")
    print(f"DDS cmd topic:   {TOPIC_CMD}")
    print(f"DDS state topic: {TOPIC_STATE}")
    print()
    print("Motor layout (per hand):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  DDS idx {i}: {name}")
    print()
    print("Values: 0.0 = CLOSED, 1.0 = OPEN")
    print("=" * 60)

    # Initialize DDS
    ChannelFactoryInitialize(0)

    pub = ChannelPublisher(TOPIC_CMD, MotorCmds_)
    pub.Init()

    sub = ChannelSubscriber(TOPIC_STATE, MotorStates_)
    sub.Init(state_callback, 1)

    time.sleep(0.5)  # Wait for subscriber to connect

    # Start fully open
    right_q = [1.0] * 6
    left_q = [1.0] * 6

    print("\n[1] Opening all fingers (q=1.0)...")
    send_and_hold(pub, right_q, left_q, duration=2.0)
    print_state()

    if args.finger >= 0 and args.finger < 6:
        # Test single finger
        finger_idx = args.finger
        print(f"\n[2] Testing single finger: {JOINT_NAMES[finger_idx]} (DDS idx {finger_idx})")
        for cycle in range(args.cycles):
            print(f"\n  --- Cycle {cycle + 1}/{args.cycles} ---")

            # Close this finger
            target_r = [1.0] * 6
            target_l = [1.0] * 6
            if test_right:
                target_r[finger_idx] = 0.0
            if test_left:
                target_l[finger_idx] = 0.0
            print(f"  Closing {JOINT_NAMES[finger_idx]}...")
            right_q, left_q = ramp_to(pub, target_r, target_l, right_q, left_q, duration=args.speed)
            send_and_hold(pub, right_q, left_q, duration=1.0)
            print_state()

            # Open this finger
            target_r = [1.0] * 6
            target_l = [1.0] * 6
            print(f"  Opening {JOINT_NAMES[finger_idx]}...")
            right_q, left_q = ramp_to(pub, target_r, target_l, right_q, left_q, duration=args.speed)
            send_and_hold(pub, right_q, left_q, duration=1.0)
            print_state()
    else:
        # Test each finger individually
        print("\n[2] Testing each finger individually...")
        for finger_idx in range(6):
            name = JOINT_NAMES[finger_idx]
            print(f"\n  --- {name} (DDS idx {finger_idx}) ---")

            # Close this finger only
            target_r = [1.0] * 6
            target_l = [1.0] * 6
            if test_right:
                target_r[finger_idx] = 0.0
            if test_left:
                target_l[finger_idx] = 0.0

            print(f"  Closing {name}...")
            right_q, left_q = ramp_to(pub, target_r, target_l, right_q, left_q, duration=args.speed)
            send_and_hold(pub, right_q, left_q, duration=1.5)
            print_state()

            # Open back
            target_r = [1.0] * 6
            target_l = [1.0] * 6
            print(f"  Opening {name}...")
            right_q, left_q = ramp_to(pub, target_r, target_l, right_q, left_q, duration=args.speed)
            send_and_hold(pub, right_q, left_q, duration=0.5)
            print_state()

        # Test all fingers together
        print("\n[3] Full open/close cycles...")
        for cycle in range(args.cycles):
            print(f"\n  --- Cycle {cycle + 1}/{args.cycles} ---")

            # Close all
            target_r = [0.0] * 6 if test_right else [1.0] * 6
            target_l = [0.0] * 6 if test_left else [1.0] * 6
            print("  Closing all...")
            right_q, left_q = ramp_to(pub, target_r, target_l, right_q, left_q, duration=args.speed)
            send_and_hold(pub, right_q, left_q, duration=1.5)
            print_state()

            # Open all
            target_r = [1.0] * 6
            target_l = [1.0] * 6
            print("  Opening all...")
            right_q, left_q = ramp_to(pub, target_r, target_l, right_q, left_q, duration=args.speed)
            send_and_hold(pub, right_q, left_q, duration=1.5)
            print_state()

        # Test half-close (0.5)
        print("\n[4] Half-close test (q=0.5)...")
        target_r = [0.5] * 6 if test_right else [1.0] * 6
        target_l = [0.5] * 6 if test_left else [1.0] * 6
        right_q, left_q = ramp_to(pub, target_r, target_l, right_q, left_q, duration=args.speed)
        send_and_hold(pub, right_q, left_q, duration=2.0)
        print_state()

    # End fully open
    print("\n[END] Opening all fingers...")
    target_r = [1.0] * 6
    target_l = [1.0] * 6
    right_q, left_q = ramp_to(pub, target_r, target_l, right_q, left_q, duration=args.speed)
    send_and_hold(pub, right_q, left_q, duration=1.0)
    print_state()
    print("\nDone!")


if __name__ == "__main__":
    main()
