#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Teach-and-Capture Server:
  setting.py 스타일로 로봇을 수동 조작하면서,
  원하는 위치에서 'c' 입력하면 클라이언트 카메라 4대 동시 촬영.

Commands:
  p <axis>,<value>  : TCP move (e.g., "p z,50")
  j <axis>,<value>  : Joint move (e.g., "j d1,10")
  show              : Show current TCP pose & joints
  speed <0-100>     : Set speed override
  c                 : Capture! (send capture command to client)
  q                 : Quit

Usage:
  [Robot]    python teach_and_capture.py
  [Computer] python new2_Step2_capture_cube_poses.py \
               --root_folder ./data/session_manual \
               --intrinsics_dir ./intrinsics \
               --use_robot --manual_robot \
               --robot_ip 192.168.0.23 --robot_port 12348 \
               --save_depth --show --min_markers 1 --min_cams_with_cube 1
"""

from i611_MCS import *
from teachdata import *
from i611_extend import *
from rbsys import *
from i611_common import *
from i611_io import *
from i611shm import *
import time
import socket
import json

HOST = '0.0.0.0'
PORT = 12348


# ──────────────────────────────────────────────────────────────
# Socket helpers
# ──────────────────────────────────────────────────────────────

def send_json(conn, obj):
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
            return None
        return json.loads(data.strip())
    except Exception as e:
        print "Recv error: {}".format(e)
    return None


# ──────────────────────────────────────────────────────────────
# Robot helpers
# ──────────────────────────────────────────────────────────────

def get_tcp():
    pose = rb.getpos()
    vals = pose.pos2list()
    return [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]]


def get_joints():
    jnt = rb.getjnt()
    vals = jnt.jnt2list()
    return [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]]


def show_pose():
    tcp = get_tcp()
    jnt = get_joints()
    print ''
    print '=== Current TCP Pose ==='
    print '  x={:.3f}  y={:.3f}  z={:.3f}'.format(tcp[0], tcp[1], tcp[2])
    print '  rz={:.3f}  ry={:.3f}  rx={:.3f}'.format(tcp[3], tcp[4], tcp[5])
    print '=== Current Joints ==='
    print '  d1={:.3f}  d2={:.3f}  d3={:.3f}'.format(jnt[0], jnt[1], jnt[2])
    print '  d4={:.3f}  d5={:.3f}  d6={:.3f}'.format(jnt[3], jnt[4], jnt[5])
    print ''
    return tcp


def move_tcp(axis, value):
    pose = rb.getpos()
    vals = pose.pos2list()
    current = Position(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])

    axis_map = {
        'x': 'dx', 'y': 'dy', 'z': 'dz',
        'rz': 'drz', 'ry': 'dry', 'rx': 'drx'
    }
    if axis not in axis_map:
        print 'Invalid axis: {}. Use x,y,z,rz,ry,rx'.format(axis)
        return

    kwargs = {axis_map[axis]: value}
    target = current.offset(**kwargs)
    print 'TCP move: {} += {}'.format(axis, value)
    rb.line(target)
    print 'Move complete'


def move_joint(axis, value):
    jnt = rb.getjnt()
    vals = jnt.jnt2list()
    current = Joint(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])

    axis_map = {
        'd1': 'dj1', 'd2': 'dj2', 'd3': 'dj3',
        'd4': 'dj4', 'd5': 'dj5', 'd6': 'dj6'
    }
    if axis not in axis_map:
        print 'Invalid axis: {}. Use d1,d2,d3,d4,d5,d6'.format(axis)
        return

    kwargs = {axis_map[axis]: value}
    target = current.offset(**kwargs)
    print 'Joint move: {} += {}'.format(axis, value)
    rb.move(target)
    print 'Move complete'


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    try:
        rbs = RobSys()
        rbs.open()

        global rb
        rb = i611Robot()
        _BASE = Base()
        rb.open()
        IOinit(rb)

        m = MotionParam(jnt_speed=30, lin_speed=50, pose_speed=50,
                        overlap=0, acctime=0.8, dacctime=0.8)
        rb.motionparam(m)
        rb.override(30)

        rb.settool(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rb.settool(2, 0.0, 35.0, 330.0, 0.0, 0.0, 0.0)
        rb.settool(3, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0)
        rb.changetool(3)
        rb.use_mt(True)

        # ─── Start server ───
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print "Teach-and-Capture server on port {}. Waiting for client...".format(PORT)

        conn, addr = s.accept()
        print "Client connected: {}".format(addr)

        print ''
        print '======================================'
        print '  Teach-and-Capture Mode'
        print '======================================'
        print 'Commands:'
        print '  p <axis>,<value>  : TCP move (p z,50)'
        print '  j <axis>,<value>  : Joint move (j d1,10)'
        print '  show              : Show current pose'
        print '  speed <0-100>     : Set speed'
        print '  c                 : ** CAPTURE ** (trigger all cameras)'
        print '  q                 : Quit'
        print '======================================'
        print ''

        show_pose()

        capture_count = 0

        while True:
            try:
                cmd = raw_input('> ').strip()
            except EOFError:
                break

            if not cmd:
                continue

            # ─── Quit ───
            if cmd.lower() == 'q':
                send_json(conn, {"command": "quit"})
                break

            # ─── Show pose ───
            elif cmd.lower() == 'show':
                show_pose()

            # ─── Speed ───
            elif cmd.lower().startswith('speed'):
                try:
                    spd = int(cmd.split()[1])
                    rb.override(spd)
                    print 'Speed set to {}'.format(spd)
                except Exception:
                    print 'Usage: speed <0-100>'

            # ─── CAPTURE ───
            elif cmd.lower() == 'c':
                tcp = get_tcp()
                print ''
                print '*** CAPTURING *** (pose_index={})'.format(capture_count)
                print '  TCP: [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                    tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])

                send_json(conn, {
                    "command": "capture",
                    "capture_pose_6dof": tcp,
                    "pose_index": capture_count
                })

                # Wait for client to finish
                resp = recv_json(conn)
                if resp is None:
                    print 'Client disconnected!'
                    break

                status = resp.get('status', 'unknown') if isinstance(resp, dict) else 'unknown'
                print '*** Capture {} done (status={}) ***'.format(capture_count, status)
                print ''
                capture_count += 1

            # ─── TCP move ───
            elif cmd.lower().startswith('p '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis = parts[0].strip()
                    value = float(parts[1].strip())
                    move_tcp(axis, value)
                    show_pose()
                except Exception as e:
                    print 'Error: {}. Usage: p <axis>,<value>'.format(e)

            # ─── Joint move ───
            elif cmd.lower().startswith('j '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis = parts[0].strip()
                    value = float(parts[1].strip())
                    move_joint(axis, value)
                    show_pose()
                except Exception as e:
                    print 'Error: {}. Usage: j <axis>,<value>'.format(e)

            else:
                print 'Unknown: {}. (p/j/c/show/speed/q)'.format(cmd)

        print '\nTotal captures: {}'.format(capture_count)

    except KeyboardInterrupt:
        print '\nInterrupted'
        try:
            send_json(conn, {"command": "quit"})
        except Exception:
            pass

    except Robot_emo as e:
        print(e)
    except Robot_error as e:
        print(e)
    except Robot_fatalerror as e:
        print(e)
    except Exception as e:
        print(e)

    finally:
        try:
            rb.exit(0)
            rb.close()
            rbs.close()
        except Exception:
            pass
        try:
            s.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
