# new2_robot_calb_server.py
"""
Zeus Robot Calibration Server v2 (i611 SDK)
Place-and-Capture workflow for multi-camera + gripper camera calibration.

Workflow (per waypoint):
  1. Server -> Client: {"command": "ready"}
  2. Client -> Server: {"action": "waypoint", "place_pose": [6], "capture_pose": [6]}
  3. Server: move to capture_pose (approach height)
  4. Server: move down to place_pose
  5. Server: open gripper (release cube)
  6. Server: move up to capture_pose
  7. Server -> Client: {"command": "capture", "capture_pose_6dof": [...], "place_pose_6dof": [...]}
  8. Client captures from all cameras
  9. Client -> Server: {"action": "captured"}
 10. Server: move down to place_pose
 11. Server: close gripper (grab cube)
 12. Server: move up to capture_pose (retreat)
 13. Repeat from 1

Usage:
  python new2_robot_calb_server.py
"""

from i611_extend import *
from rbsys import *
from i611_common import *
from i611_io import *
from i611shm import *
import time
import socket
import json

# ─── Configuration ───
HOST = '0.0.0.0'
PORT = 12348

# Gripper IO ports (adjust to your hardware)
GRIPPER_OPEN_PORT = 1    # DO port number for gripper open
GRIPPER_CLOSE_PORT = 2   # DO port number for gripper close
GRIPPER_SETTLE_SEC = 0.8 # wait time after gripper action

# Set True if no automatic gripper (server will pause and print message)
MANUAL_GRIPPER = True
MANUAL_GRIPPER_WAIT_SEC = 3.0

# Move speed override (0-100)
SPEED_OVERRIDE = 50


# ──────────────────────────────────────────────────────────────
# Socket helpers
# ──────────────────────────────────────────────────────────────

def send_json(conn, obj):
    if conn is None:
        print "[ERROR] Connection is None"
        return
    try:
        msg = json.dumps(obj)
        conn.sendall(msg.encode('utf-8'))
        print "Sent: {}".format(msg)
    except socket.error as e:
        print "Send error: {}".format(e)


def recv_json(conn):
    try:
        data = conn.recv(8192).decode('utf-8')
        if not data:
            print "[WARN] Empty data received"
            return None
        data = data.strip()
        try:
            obj = json.loads(data)
            print "Recv: {}".format(obj)
            return obj
        except ValueError as e:
            print "JSON error: {} | raw: {}".format(e, data)
            return None
    except socket.error as e:
        print "Recv error: {}".format(e)
    return None


# ──────────────────────────────────────────────────────────────
# Robot helpers
# ──────────────────────────────────────────────────────────────

def get_tcp_pose():
    """Current TCP pose as [x, y, z, rz, ry, rx]."""
    pose = rb.getpos()
    vals = pose.pos2list()
    return [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]]


def move_to_pos(pose_6dof):
    """Move robot to a TCP Cartesian position."""
    goal = Position(pose_6dof[0], pose_6dof[1], pose_6dof[2],
                    pose_6dof[3], pose_6dof[4], pose_6dof[5])
    print 'Moving to: [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
        pose_6dof[0], pose_6dof[1], pose_6dof[2],
        pose_6dof[3], pose_6dof[4], pose_6dof[5])
    rb.line(goal)
    print 'Move complete'


def gripper_open():
    if MANUAL_GRIPPER:
        print '>>> MANUAL: Release the cube, then wait...'
        time.sleep(MANUAL_GRIPPER_WAIT_SEC)
    else:
        print 'Gripper opening...'
        DOWrite(GRIPPER_OPEN_PORT, 1)
        DOWrite(GRIPPER_CLOSE_PORT, 0)
        time.sleep(GRIPPER_SETTLE_SEC)
    print 'Gripper opened'


def gripper_close():
    if MANUAL_GRIPPER:
        print '>>> MANUAL: Grab the cube, then wait...'
        time.sleep(MANUAL_GRIPPER_WAIT_SEC)
    else:
        print 'Gripper closing...'
        DOWrite(GRIPPER_OPEN_PORT, 0)
        DOWrite(GRIPPER_CLOSE_PORT, 1)
        time.sleep(GRIPPER_SETTLE_SEC)
    print 'Gripper closed'


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────

def main(conn):
    try:
        m = MotionParam(jnt_speed=70, lin_speed=50, pose_speed=50,
                        overlap=0, acctime=1.0, dacctime=1.0)
        rb.motionparam(m)
        rb.override(SPEED_OVERRIDE)

        rb.settool(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rb.settool(2, 0.0, 35.0, 330.0, 0.0, 0.0, 0.0)
        rb.settool(3, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0)
        rb.changetool(3)
        rb.use_mt(True)

        pose_index = 0

        while True:
            print '\n======== Waypoint {} ========'.format(pose_index)

            # 1. Signal ready
            send_json(conn, {"command": "ready", "pose_index": pose_index})

            # 2. Wait for waypoint from client
            msg = recv_json(conn)
            if msg is None:
                print 'Connection lost'
                break

            action = msg.get('action', '') if isinstance(msg, dict) else ''
            if action == 'quit':
                print 'Client quit'
                break

            if action != 'waypoint':
                print 'Unknown action: {}'.format(action)
                continue

            place_pose = msg.get('place_pose', None)
            capture_pose = msg.get('capture_pose', None)
            if place_pose is None or capture_pose is None:
                print 'Invalid waypoint data'
                continue

            # 3. Move to capture position first (safe approach height)
            print '--- Phase: Approach (move to capture height) ---'
            move_to_pos(capture_pose)

            # 4. Move down to place position
            print '--- Phase: Descend to place position ---'
            move_to_pos(place_pose)

            # 5. Open gripper (release cube)
            print '--- Phase: Release cube ---'
            gripper_open()

            # 6. Move up to capture position
            print '--- Phase: Ascend to capture position ---'
            move_to_pos(capture_pose)
            time.sleep(0.5)  # settle

            # 7. Read actual TCP poses and signal capture
            actual_capture_tcp = get_tcp_pose()
            actual_place_tcp = place_pose  # we already recorded this
            print 'Actual capture TCP: {}'.format(actual_capture_tcp)

            send_json(conn, {
                "command": "capture",
                "capture_pose_6dof": actual_capture_tcp,
                "place_pose_6dof": actual_place_tcp,
                "pose_index": pose_index
            })

            # 8. Wait for client to finish capturing
            resp = recv_json(conn)
            if resp is None:
                print 'Connection lost during capture wait'
                break

            resp_action = resp.get('action', '') if isinstance(resp, dict) else ''
            if resp_action == 'quit':
                print 'Client quit during capture'
                break

            # 9. Move back down to place position (pick up cube)
            print '--- Phase: Descend to pick up cube ---'
            move_to_pos(place_pose)

            # 10. Close gripper (grab cube)
            print '--- Phase: Grab cube ---'
            gripper_close()

            # 11. Move up to capture position (retreat)
            print '--- Phase: Retreat ---'
            move_to_pos(capture_pose)

            pose_index += 1
            print '======== Waypoint {} complete ========'.format(pose_index - 1)

        print '\nSession complete. Total waypoints: {}'.format(pose_index)

    except Robot_emo as e:
        print(e)
        rb.exit(0)
        rbs.cmd_reset()

    except Robot_error as e:
        print(e)
        rb.exit(0)
        rbs.cmd_reset()

    except Robot_fatalerror as e:
        print(e)
        rb.exit(0)
        rbs.cmd_reset()

    except Exception as e:
        print(e)
        rb.exit(0)

    except KeyboardInterrupt:
        rb.exit(0)
        print 'Key Interrupt'

    finally:
        rb.close()
        rbs.close()
        rb.exit(0)


# ──────────────────────────────────────────────────────────────
# Server start
# ──────────────────────────────────────────────────────────────

def start_server():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print "Server v2 started on port {}. Waiting for client...".format(PORT)

        conn, addr = s.accept()
        print "Client connected: {}".format(addr)

        main(conn)

    except socket.error as e:
        print "Socket error: {}".format(e)
    finally:
        s.close()


if __name__ == '__main__':
    try:
        rbs = RobSys()
        rbs.open()
        rb = i611Robot()
        _BASE = Base()

        rb.open()
        IOinit(rb)

        start_server()

    except Exception as e:
        print(e)
        rb.exit(0)

    except Robot_emo:
        rb.exit(0)
        rbs.cmd_reset()

    except Robot_error:
        rb.exit(0)
        rbs.cmd_reset()

    except Robot_fatalerror:
        rb.exit(0)
        rbs.cmd_reset()

    finally:
        rb.close()
        rbs.close()
        rb.exit(0)
