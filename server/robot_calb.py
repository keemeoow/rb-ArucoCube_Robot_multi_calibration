#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
로봇 캘리브레이션 서버 (Teach-and-Capture):
  수동 조작으로 로봇을 이동/회전하면서 촬영하는 서버.

명령어:
  --- 이동 ---
  p <축>,<값>       : TCP 상대 이동 (예: "p z,50", "p rz,15")
  j <축>,<값>       : 관절 상대 이동 (예: "j d1,10")
  goto x,y,z,rz,ry,rx : TCP 절대 좌표로 이동
  show              : 현재 TCP 포즈 및 관절 값 표시
  speed <0-100>     : 속도 설정 (클수록 빠름)

  --- 촬영 ---
  c                 : 현재 위치에서 촬영

  --- 설정 ---
  set               : 현재 TCP + 큐브 중점(Tool 4)을 저장
                      큐브 중점은 매 촬영 시 PC로 전송됨

  --- 그리퍼 ---
  go                : 그리퍼 열기
  gc                : 그리퍼 닫기

  --- 되돌리기 ---
  undo              : 마지막 이동 1회 되돌리기
  undo <N>          : 마지막 N회 되돌리기
  undo all          : 전체 되돌리기
  undo <axis...>    : 특정 축만 되돌리기 (undo x ry rz)
  undo set          : 어디서든 set 위치로 이동

  --- 종료 ---
  q                 : 종료

══════════════════════════════════════════════════════════════
전체 파이프라인
══════════════════════════════════════════════════════════════

Step 1 -- 카메라 내부 파라미터 (intrinsics)
--------------------------------------------------------------
  [PC]
  python Step1_dump_all_intrinsics.py \
    --out_dir ./intrinsics \
    --gripper_serial <그리퍼_카메라_시리얼>

Step 2 -- 큐브 놓으면서 촬영
--------------------------------------------------------------
  수동으로 큐브 돌출부 중점에 정확히 잡은 상태에서 "set" 저장.
  이후 이동 -> 놓기 -> 촬영 -> "undo set" 복귀 -> 잡기 반복.

  - 그리퍼 카메라: ChArUco 보드 + ArUco 큐브 검출
  - 고정 카메라: ArUco 큐브만 검출

  [로봇-서버]
  python robot_calb.py

  [PC]
  python Step2_to_capture_capture.py \
    --root_folder ./data/session \
    --intrinsics_dir ./intrinsics \
    --use_robot --manual_robot \
    --robot_ip 192.168.0.23 --robot_port 12348 \
    --show --save_depth \
    --min_markers 1 --min_cams_with_cube 1

  플로우:
    1. 수동으로 큐브 돌출부 중점 잡기
    2. "set"          -> TCP + 큐브 중점 저장
    3. 이동 -> 촬영 위치
    4. "go"           -> 큐브 놓기
    5. "c"            -> 촬영
    6. "gc"           -> 큐브 잡기
    7. 3~6 반복
    8. "undo set"     -> set 위치로 복귀

Step 3 -- 캘리브레이션
--------------------------------------------------------------
  [PC]
  python Step3_to_calibration.py \
    --root_folder ./data/session \
    --intrinsics_dir ./intrinsics

"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

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

# Gripper IO port
GRIPPER_IO_PORT = 48
GRIPPER_TIMEOUT_SEC = 5.0


# Cube grip parameters
CUBE_SIZE_MM = 30.0         # cube side length (mm)
CUBE_EDGE_MM = 2.0          # edge protrusion height (mm)
CUBE_GRIP_DEPTH_MM = 2.0    # fingertip enters 2mm below cube top
# TCP(fingertip) to cube center (along tool z): 15mm - 2mm = 13mm
CUBE_CENTER_OFFSET_Z = CUBE_SIZE_MM / 2.0 - CUBE_GRIP_DEPTH_MM

# Tool frame offsets (from robot flange)
TOOL_GRIPPER_Z = 150.0                                  # tool 3: fingertip
TOOL_CUBE_CENTER_Z = TOOL_GRIPPER_Z - CUBE_CENTER_OFFSET_Z  # tool 4: cube center (137mm)


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


def get_cube_center():
    """Read cube center TCP via tool 4 (accounts for grip depth + rotation)."""
    rb.changetool(4)
    pose = rb.getpos()
    vals = pose.pos2list()
    rb.changetool(3)
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


def check_gripper():
    """Read gripper state from din 48~51."""
    a = din(GRIPPER_IO_PORT)
    b = din(GRIPPER_IO_PORT + 1)
    c = din(GRIPPER_IO_PORT + 2)
    d = din(GRIPPER_IO_PORT + 3)
    return [d, c, b, a]


def gripper_open():
    print 'Gripper opening...'
    dout(GRIPPER_IO_PORT, '0000')
    t0 = time.time()
    while check_gripper() != ['0', '1', '0', '0']:
        dout(GRIPPER_IO_PORT, '0100')
        if time.time() - t0 > GRIPPER_TIMEOUT_SEC:
            print '[WARN] Gripper open timeout!'
            break
        time.sleep(0.05)
    print 'Gripper opened'


def gripper_close():
    print 'Gripper closing...'
    dout(GRIPPER_IO_PORT, '0000')
    t0 = time.time()
    while check_gripper() != ['0', '0', '0', '1']:
        dout(GRIPPER_IO_PORT, '0001')
        if time.time() - t0 > GRIPPER_TIMEOUT_SEC:
            print '[WARN] Gripper close timeout!'
            break
        time.sleep(0.05)
    print 'Gripper closed'


def do_capture(conn, capture_count, set_cube_center=None):
    """Send capture command and wait for response."""
    tcp = get_tcp()
    cube_tcp = get_cube_center()
    print ''
    print '*** CAPTURING *** (pose_index={})'.format(capture_count)
    print '  TCP(fingertip):   [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
        tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])
    print '  TCP(cube center): [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
        cube_tcp[0], cube_tcp[1], cube_tcp[2], cube_tcp[3], cube_tcp[4], cube_tcp[5])

    msg = {
        "command": "capture",
        "capture_pose_6dof": tcp,
        "cube_center_pose_6dof": cube_tcp,
        "pose_index": capture_count
    }
    if set_cube_center is not None:
        msg["set_cube_center_6dof"] = set_cube_center

    send_json(conn, msg)

    resp = recv_json(conn)
    if resp is None:
        print 'Client disconnected!'
        return False

    status = resp.get('status', 'unknown') if isinstance(resp, dict) else 'unknown'
    print '*** Capture {} done (status={}) ***'.format(capture_count, status)
    print ''
    return True


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

        m = MotionParam(jnt_speed=100, lin_speed=100, pose_speed=100,
                        overlap=0, acctime=0.8, dacctime=0.8)
        rb.motionparam(m)
        rb.override(100)

        rb.settool(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rb.settool(2, 0.0, 35.0, 330.0, 0.0, 0.0, 0.0)
        rb.settool(3, 0.0, 0.0, TOOL_GRIPPER_Z, 0.0, 0.0, 0.0)
        rb.settool(4, 0.0, 0.0, TOOL_CUBE_CENTER_Z, 0.0, 0.0, 0.0)
        rb.changetool(3)  # default: fingertip
        rb.use_mt(True)

        # ─── Start server ───
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print "Teach-and-Capture server on port {}. Waiting for client...".format(PORT)

        conn, addr = s.accept()
        print "Client connected: {}".format(addr)

        # ─── State ───
        capture_count = 0
        move_history = []       # [(type, axis, value), ...] for undo
        holding_cube = True     # True: cube in gripper, False: cube placed
        home_pose = None        # saved TCP from 'set'
        set_cube_center = None  # cube center 6dof from 'set' (sent to PC)
        waypoints = []          # [(tcp_6dof, cube_center_6dof), ...] saved on quit

        print ''
        print '=========================================='
        print '  Teach-and-Capture Mode'
        print '=========================================='
        print ''
        print '--- Movement ---'
        print '  p <axis>,<value>  : TCP move (p z,-10)'
        print '  j <axis>,<value>  : Joint move (j d1,10)'
        print '  show              : Show current pose'
        print '  speed <0-100>     : Set speed'
        print ''
        print '--- Capture ---'
        print '  c                 : Capture at current position'
        print ''
        print '--- Settings ---'
        print '  set               : Save TCP + cube center (Tool 4)'
        print ''
        print '--- Gripper ---'
        print '  go                : Gripper open'
        print '  gc                : Gripper close'
        print ''
        print '--- Undo ---'
        print '  undo              : Reverse last move'
        print '  undo <N>          : Reverse last N moves'
        print '  undo all          : Reverse all moves'
        print '  undo <axis...>    : Reverse axis moves (undo x ry rz)'
        print '  undo set          : Return to set position'
        print ''
        print '  q                 : Quit'
        print '=========================================='
        print ''

        show_pose()

        while True:
            try:
                cmd = raw_input('> ').strip()
            except EOFError:
                break

            if not cmd:
                continue

            cmd_lower = cmd.lower()

            # ─── Quit ───
            if cmd_lower == 'q':
                send_json(conn, {"command": "quit"})
                break

            # ─── Show pose ───
            elif cmd_lower == 'show':
                show_pose()
                if home_pose is not None:
                    print '  [Set] TCP:  [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                        home_pose[0], home_pose[1], home_pose[2],
                        home_pose[3], home_pose[4], home_pose[5])
                if set_cube_center is not None:
                    print '  [Set] Cube: [{:.1f}, {:.1f}, {:.1f}]'.format(
                        set_cube_center[0], set_cube_center[1], set_cube_center[2])
                print ''

            # ─── Speed ───
            elif cmd_lower.startswith('speed'):
                try:
                    spd = int(cmd.split()[1])
                    rb.override(spd)
                    print 'Speed set to {}'.format(spd)
                except Exception:
                    print 'Usage: speed <0-100>'

            # ─── Set: save TCP + cube center ───
            elif cmd_lower == 'set':
                home_pose = get_tcp()
                set_cube_center = get_cube_center()
                move_history = []
                print ''
                print '*** Set saved ***'
                print '  TCP:         [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                    home_pose[0], home_pose[1], home_pose[2],
                    home_pose[3], home_pose[4], home_pose[5])
                print '  Cube center: [{:.1f}, {:.1f}, {:.1f}] (offset={:.0f}mm)'.format(
                    set_cube_center[0], set_cube_center[1], set_cube_center[2],
                    CUBE_CENTER_OFFSET_Z)
                print ''

            # ─── Gripper open ───
            elif cmd_lower == 'go':
                gripper_open()
                holding_cube = False

            # ─── Gripper close ───
            elif cmd_lower == 'gc':
                gripper_close()
                holding_cube = True

            # ─── CAPTURE ───
            elif cmd_lower == 'c':
                ok = do_capture(conn, capture_count, set_cube_center)
                if not ok:
                    break
                tcp = get_tcp()
                cube_tcp = get_cube_center()
                waypoints.append({
                    "pose_index": capture_count,
                    "tcp_6dof": tcp,
                    "cube_center_6dof": cube_tcp
                })
                capture_count += 1

            # ─── UNDO ───
            elif cmd_lower.startswith('undo'):
                parts = cmd.lower().split()
                args_list = parts[1:]

                # undo set: return to saved TCP
                if len(args_list) == 1 and args_list[0] == 'set':
                    if home_pose is None:
                        print 'No set saved. Use "set" first.'
                    else:
                        tcp = get_tcp()
                        target = Position(home_pose[0], home_pose[1], home_pose[2],
                                          home_pose[3], home_pose[4], home_pose[5])
                        print ''
                        print '--- Returning to set position ---'
                        print '  from: [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                            tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])
                        print '  to:   [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                            home_pose[0], home_pose[1], home_pose[2],
                            home_pose[3], home_pose[4], home_pose[5])
                        rb.line(target)
                        move_history = []
                        print '--- Set position reached ---'
                        show_pose()

                elif not move_history:
                    print 'Nothing to undo.'
                else:
                    valid_axes = set(['x', 'y', 'z', 'rx', 'ry', 'rz',
                                      'd1', 'd2', 'd3', 'd4', 'd5', 'd6'])

                    if len(args_list) == 0:
                        last = move_history.pop()
                        mtype, maxis, mvalue = last
                        reverse = -mvalue
                        print ''
                        print '--- Undoing 1 move ---'
                        print '  {} {},{} -> {}'.format(mtype, maxis, mvalue, reverse)
                        if mtype == 'p':
                            move_tcp(maxis, reverse)
                        elif mtype == 'j':
                            move_joint(maxis, reverse)

                    elif args_list[0] == 'all':
                        count = len(move_history)
                        print ''
                        print '--- Undoing ALL {} move(s) ---'.format(count)
                        for i in range(count):
                            last = move_history.pop()
                            mtype, maxis, mvalue = last
                            reverse = -mvalue
                            print '  [{}/{}] {} {},{} -> {}'.format(
                                i + 1, count, mtype, maxis, mvalue, reverse)
                            if mtype == 'p':
                                move_tcp(maxis, reverse)
                            elif mtype == 'j':
                                move_joint(maxis, reverse)

                    elif args_list[0] in valid_axes:
                        axis_set = set([a for a in args_list if a in valid_axes])
                        targets = []
                        for i in range(len(move_history)):
                            h = move_history[i]
                            if h[1] in axis_set:
                                targets.append((i, h))
                        if not targets:
                            print 'No [{}] moves to undo.'.format(','.join(sorted(axis_set)))
                        else:
                            label = ','.join(sorted(axis_set))
                            print ''
                            print '--- Undoing {} move(s) on [{}] ---'.format(
                                len(targets), label)
                            for idx in range(len(targets) - 1, -1, -1):
                                item = targets[idx]
                                mtype, maxis, mvalue = item[1]
                                reverse = -mvalue
                                step = len(targets) - idx
                                print '  [{}/{}] {} {},{} -> {}'.format(
                                    step, len(targets), mtype, maxis, mvalue, reverse)
                                if mtype == 'p':
                                    move_tcp(maxis, reverse)
                                elif mtype == 'j':
                                    move_joint(maxis, reverse)
                                move_history.pop(item[0])

                    else:
                        try:
                            count = int(args_list[0])
                        except ValueError:
                            print 'Usage: undo / undo <N> / undo all / undo <axis...> / undo set'
                            continue
                        count = min(count, len(move_history))
                        print ''
                        print '--- Undoing {} move(s) ---'.format(count)
                        for i in range(count):
                            last = move_history.pop()
                            mtype, maxis, mvalue = last
                            reverse = -mvalue
                            print '  [{}/{}] {} {},{} -> {}'.format(
                                i + 1, count, mtype, maxis, mvalue, reverse)
                            if mtype == 'p':
                                move_tcp(maxis, reverse)
                            elif mtype == 'j':
                                move_joint(maxis, reverse)

                    print '--- Undo complete ---'
                    show_pose()

            # ─── GOTO: absolute TCP position ───
            elif cmd_lower.startswith('goto '):
                try:
                    vals = [float(v.strip()) for v in cmd[5:].strip().split(',')]
                    if len(vals) == 6:
                        target = Position(vals[0], vals[1], vals[2],
                                          vals[3], vals[4], vals[5])
                        print 'GOTO: [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
                        rb.line(target)
                        print 'Move complete'
                        show_pose()
                    elif len(vals) == 3:
                        tcp = get_tcp()
                        target = Position(vals[0], vals[1], vals[2],
                                          tcp[3], tcp[4], tcp[5])
                        print 'GOTO: [{:.1f}, {:.1f}, {:.1f}] (rotation unchanged)'.format(
                            vals[0], vals[1], vals[2])
                        rb.line(target)
                        print 'Move complete'
                        show_pose()
                    else:
                        print 'Usage: goto x,y,z  or  goto x,y,z,rz,ry,rx'
                except Exception as e:
                    print 'Error: {}. Usage: goto x,y,z,rz,ry,rx'.format(e)

            # ─── TCP move ───
            elif cmd_lower.startswith('p '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis = parts[0].strip()
                    value = float(parts[1].strip())
                    move_tcp(axis, value)
                    move_history.append(('p', axis, value))
                    show_pose()
                except Exception as e:
                    print 'Error: {}. Usage: p <axis>,<value>'.format(e)

            # ─── Joint move ───
            elif cmd_lower.startswith('j '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis = parts[0].strip()
                    value = float(parts[1].strip())
                    move_joint(axis, value)
                    move_history.append(('j', axis, value))
                    show_pose()
                except Exception as e:
                    print 'Error: {}. Usage: j <axis>,<value>'.format(e)

            else:
                print 'Unknown: {}'.format(cmd)

        # Save waypoints for next session
        if waypoints:
            wp_path = 'capture_waypoints.json'
            with open(wp_path, 'w') as f:
                json.dump(waypoints, f, indent=2)
            print '\nWaypoints saved: {} ({} poses)'.format(wp_path, len(waypoints))

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