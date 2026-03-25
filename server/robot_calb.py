"""
로봇 캘리브레이션 서버 (Teach-and-Capture):
  수동 조작으로 로봇을 이동/회전하면서 촬영하는 서버.

명령어:
  --- 이동 ---
  p <축>,<값>       : TCP 이동 (예: "p z,50", "p rz,15")
  j <축>,<값>       : 관절 이동 (예: "j d1,10")
  show              : 현재 TCP 포즈 및 관절 값 표시
  speed <0-100>     : 속도 설정 (클수록 빠름)

  --- 촬영 ---
  scan              : 자동 사이클: 놓기 -> 촬영 -> 집기
  scan ry,15 rz,-20 : 자동 사이클 + 회전 촬영 추가
  c                 : 현재 위치에서 촬영만
  co                : 큐브 놓기 (그리퍼 열기 -> z +22mm)
  cc                : 큐브 집기 (z -22mm -> 그리퍼 닫기)

  --- 설정 ---
  set               : 현재 TCP를 홈(큐브 잡는) 포즈로 저장

  --- 그리퍼 ---
  go                : 그리퍼 열기
  gc                : 그리퍼 닫기

  --- 되돌리기 ---
  undo              : 마지막 이동 1회 되돌리기
  undo <N>          : 마지막 N회 되돌리기
  undo all          : 전체 되돌리기

  --- 종료 ---
  q                 : 종료

══════════════════════════════════════════════════════════════
전체 파이프라인
══════════════════════════════════════════════════════════════

Step 1 — 카메라 내부 파라미터 (intrinsics)
──────────────────────────────────────────────────────────────
  python Step1_dump_all_intrinsics.py \
    --out_dir ./intrinsics \
    --gripper_serial <그리퍼_카메라_시리얼>

Step 2-1 — Board Zone: hand-in-eye (그리퍼 카메라 → 보드)
──────────────────────────────────────────────────────────────
  그리퍼 카메라로 ChArUco 보드를 다양한 각도에서 촬영.
  → T_gripper_cam 추정용 데이터 수집.

  플로우:
    1. 큐브 없이, 그리퍼를 보드 위 다양한 자세로 이동
    2. "c" → 촬영 (그리퍼 카메라가 보드 촬영)
    3. "p ry,15" 등으로 자세 변경 → "c" 반복

  [로봇]    python robot_calb.py
  [컴퓨터]  python Step2_in_capture_capture.py \
              --root_folder ./data/board_session \
              --intrinsics_dir ./intrinsics \
              --gripper_cam_idx 2 \
              --robot_ip 192.168.0.23 --robot_port 12348 \
              --show

Step 2-2a — Cube Zone: hand-to-eye (큐브 쥔 채로 촬영)
──────────────────────────────────────────────────────────────
  큐브를 그리퍼로 쥔 상태에서 다양한 자세로 이동하며 촬영.
  고정 카메라들이 큐브의 여러 면을 관측 → T_base_fixedcam 추정.

  플로우:
    1. 큐브를 쥐고 원하는 위치로 이동
    2. "c" → 촬영 (쥔 채로, 고정 카메라가 큐브 관측)
    3. "p rz,30" 등으로 자세 변경 → "c" 반복
    4. 다른 위치로 이동 후 반복

  [로봇]    python robot_calb.py
  [컴퓨터]  python Step2_to_capture_capture.py \
              --root_folder ./data/cube_session \
              --intrinsics_dir ./intrinsics \
              --manual_robot \
              --robot_ip 192.168.0.23 --robot_port 12348 \
              --show

Step 2-2b — Bridge Zone: 연결 (보드 옆에 큐브 놓고 촬영)
──────────────────────────────────────────────────────────────
  큐브를 보드 옆에 놓고 그리퍼 카메라를 높이 올려서
  보드 + 큐브를 동시 검출 → T_board_cube 획득.
  고정 카메라도 큐브를 관측 → 두 캘리브레이션을 연결하는 구속 조건.

  플로우:
    1. 큐브를 잡고 보드 옆 위치로 이동
    2. "scan" → 자동으로 놓기/촬영/집기
       (그리퍼 카메라가 보드+큐브 동시 촬영, 고정 카메라가 큐브 촬영)
    3. 다른 위치로 이동 후 반복

  [로봇]    python robot_calb.py
  [컴퓨터]  python Step2_in_capture_capture.py \
              --root_folder ./data/bridge_session \
              --intrinsics_dir ./intrinsics \
              --gripper_cam_idx 2 \
              --robot_ip 192.168.0.23 --robot_port 12348 \
              --show --also_detect_cube

Step 3-1 — Hand-in-eye 캘리브레이션
──────────────────────────────────────────────────────────────
  python Step3_in_calibration.py \
    --charuco_folder ./data/board_session \
    --intrinsics_dir ./intrinsics \
    --gripper_cam_idx 2

Step 3-2 — Hand-to-eye 캘리브레이션
──────────────────────────────────────────────────────────────
  python Step3_to_calibration.py \
    --root_folder ./data/cube_session \
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
CUBE_GRIP_Z_ABOVE = 2.0    # grip 2mm above cube top
# co/cc lift height: cube half (15) + grip offset (2) + margin (5) = 22mm
CUBE_LIFT_Z = 22.0


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


def move_z_to(target_z):
    """Move only z axis to absolute target_z value."""
    tcp = get_tcp()
    current = Position(tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])
    dz = target_z - tcp[2]
    target = current.offset(dz=dz)
    print 'Moving z: {:.1f} -> {:.1f} (dz={:.1f})'.format(tcp[2], target_z, dz)
    rb.line(target)
    print 'Move complete'


def move_z_offset(offset):
    """Move z axis by offset mm."""
    tcp = get_tcp()
    current = Position(tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])
    target = current.offset(dz=offset)
    print 'Moving z: {:.1f} -> {:.1f} (dz={:.1f})'.format(tcp[2], tcp[2] + offset, offset)
    rb.line(target)
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


def do_capture(conn, capture_count, cube_center_6dof=None):
    """Send capture command and wait for response."""
    tcp = get_tcp()
    print ''
    print '*** CAPTURING *** (pose_index={})'.format(capture_count)
    print '  TCP: [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
        tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])
    if cube_center_6dof is not None:
        print '  Cube center: [{:.1f}, {:.1f}, {:.1f}]'.format(
            cube_center_6dof[0], cube_center_6dof[1], cube_center_6dof[2])

    msg = {
        "command": "capture",
        "capture_pose_6dof": tcp,
        "pose_index": capture_count
    }
    if cube_center_6dof is not None:
        msg["cube_center_6dof"] = cube_center_6dof

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

        # ─── State ───
        capture_count = 0
        move_history = []       # [(type, axis, value), ...] for undo
        cube_center_6dof = None # cube center position when placed (grip corrected)
        holding_cube = True     # True: cube in gripper, False: cube placed
        home_pose = None        # saved TCP pose from 'set' command

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
        print '--- Phase 2a: Cube Zone (holding cube) ---'
        print '  c                 : Capture (cube held -> auto compute center)'
        print ''
        print '--- Phase 2b: Bridge Zone (place cube near board) ---'
        print '  scan              : Place -> capture -> pickup'
        print '  scan ry,15 rz,-20 : Place -> capture -> rotate+capture -> pickup'
        print '  co                : Place cube (open gripper -> z +22mm)'
        print '  cc                : Pickup cube (z -22mm -> close gripper)'
        print ''
        print '--- Settings ---'
        print '  set               : Save current TCP as home (cube grip) pose'
        print ''
        print '--- Gripper / Undo ---'
        print '  go / gc           : Gripper open / close'
        print '  undo [N|all]      : Reverse last move(s)'
        print '  undo <axis...>    : Reverse axis moves (undo x ry rz)'
        print '  undo set          : Return to saved home pose'
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
                    print '  [Home] [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                        home_pose[0], home_pose[1], home_pose[2],
                        home_pose[3], home_pose[4], home_pose[5])
                print ''

            # ─── Speed ───
            elif cmd_lower.startswith('speed'):
                try:
                    spd = int(cmd.split()[1])
                    rb.override(spd)
                    print 'Speed set to {}'.format(spd)
                except Exception:
                    print 'Usage: speed <0-100>'

            # ─── Set home pose ───
            elif cmd_lower == 'set':
                home_pose = get_tcp()
                move_history = []
                print ''
                print '*** Home pose saved ***'
                print '  [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                    home_pose[0], home_pose[1], home_pose[2],
                    home_pose[3], home_pose[4], home_pose[5])
                print ''

            # ─── Gripper open ───
            elif cmd_lower == 'go':
                gripper_open()
                holding_cube = False

            # ─── Gripper close ───
            elif cmd_lower == 'gc':
                gripper_close()
                holding_cube = True

            # ─── SCAN: Phase 2b auto cycle with rotations ───
            # scan                    : co → c → cc (1 capture, no rotation)
            # scan ry,15              : co → c → ry+15 → c → undo → cc
            # scan ry,15 rz,-20      : co → c → ry+15 → c → undo → rz-20 → c → undo → cc
            elif cmd_lower.startswith('scan'):
                # Parse rotation arguments
                rotations = []
                parts = cmd.split()
                for part in parts[1:]:
                    try:
                        ax, val = part.split(',')
                        rotations.append((ax.strip(), float(val.strip())))
                    except Exception:
                        print 'Invalid rotation: {}. Format: axis,value'.format(part)

                # --- CO: Place cube ---
                grip_tcp = get_tcp()
                grip_offset_z = CUBE_GRIP_Z_ABOVE + CUBE_SIZE_MM / 2.0
                cube_center_6dof = list(grip_tcp)
                cube_center_6dof[2] = grip_tcp[2] - grip_offset_z

                n_shots = 1 + len(rotations)
                print ''
                print '====== SCAN (captures={}) ======'.format(n_shots)
                print '  Cube center: [{:.1f}, {:.1f}, {:.1f}]'.format(
                    cube_center_6dof[0], cube_center_6dof[1], cube_center_6dof[2])
                if rotations:
                    print '  Rotations: {}'.format(
                        ['{}={}'.format(a, v) for a, v in rotations])

                gripper_open()
                holding_cube = False
                move_z_offset(CUBE_LIFT_Z)

                # --- Base capture (no rotation) ---
                print '--- Shot 1/{}: base ---'.format(n_shots)
                ok = do_capture(conn, capture_count, cube_center_6dof)
                if not ok:
                    break
                capture_count += 1

                # --- Rotated captures ---
                for i, (ax, val) in enumerate(rotations):
                    print '--- Shot {}/{}: {} {} ---'.format(i + 2, n_shots, ax, val)
                    move_tcp(ax, val)
                    ok = do_capture(conn, capture_count, cube_center_6dof)
                    if not ok:
                        break
                    capture_count += 1
                    # Undo this rotation
                    move_tcp(ax, -val)
                else:
                    ok = True

                if not ok:
                    break

                # --- CC: Pickup cube ---
                move_z_offset(-CUBE_LIFT_Z)
                gripper_close()
                holding_cube = True

                print '====== SCAN DONE ({} captures) ======'.format(n_shots)
                print '  Total: {}'.format(capture_count)
                print ''

            # ─── CAPTURE only (no move/gripper) ───
            elif cmd_lower == 'c':
                if holding_cube:
                    # Phase 2a: compute cube center from current grip TCP
                    grip_tcp = get_tcp()
                    grip_offset_z = CUBE_GRIP_Z_ABOVE + CUBE_SIZE_MM / 2.0
                    cube_center_6dof = list(grip_tcp)
                    cube_center_6dof[2] = grip_tcp[2] - grip_offset_z
                    print '  [Holding] cube center: [{:.1f}, {:.1f}, {:.1f}]'.format(
                        cube_center_6dof[0], cube_center_6dof[1], cube_center_6dof[2])
                ok = do_capture(conn, capture_count, cube_center_6dof)
                if not ok:
                    break
                capture_count += 1

            # ─── CO: Place cube (open gripper -> z up 22mm) ───
            elif cmd_lower == 'co':
                print ''
                print '--- CO: Place cube ---'

                # Record grip TCP BEFORE opening (cube is still held)
                grip_tcp = get_tcp()
                # Cube center = grip point - 2mm (above top) - 15mm (half cube)
                grip_offset_z = CUBE_GRIP_Z_ABOVE + CUBE_SIZE_MM / 2.0  # 17mm
                cube_center_6dof = list(grip_tcp)
                cube_center_6dof[2] = grip_tcp[2] - grip_offset_z
                print '  Grip TCP z:     {:.1f}'.format(grip_tcp[2])
                print '  Cube center z:  {:.1f} (TCP - {:.0f}mm)'.format(
                    cube_center_6dof[2], grip_offset_z)

                print '--- 1/2: Gripper open ---'
                gripper_open()
                holding_cube = False
                move_history = []  # reset undo history after placing
                print '--- 2/2: Moving z +{:.0f}mm ---'.format(CUBE_LIFT_Z)
                move_z_offset(CUBE_LIFT_Z)
                print '--- CO done: cube placed, gripper above ---'
                print ''

            # ─── CC: Pickup cube (z down 22mm -> close gripper) ───
            elif cmd_lower == 'cc':
                print ''
                print '--- CC: Pickup cube ---'
                print '--- 1/2: Moving z -{:.0f}mm ---'.format(CUBE_LIFT_Z)
                move_z_offset(-CUBE_LIFT_Z)
                print '--- 2/2: Gripper close ---'
                gripper_close()
                holding_cube = True
                print '--- CC done: cube grabbed ---'
                print ''

            # ─── UNDO: reverse moves ───
            # undo              : reverse last 1 move
            # undo 3            : reverse last 3 moves
            # undo all          : reverse ALL moves
            # undo x            : reverse all x moves
            # undo x ry rz      : reverse all x, ry, rz moves
            elif cmd_lower.startswith('undo'):
                parts = cmd.lower().split()
                args_list = parts[1:]

                # ─── undo set: return to saved home pose ───
                if len(args_list) == 1 and args_list[0] == 'set':
                    if home_pose is None:
                        print 'No home pose saved. Use "set" first.'
                    else:
                        tcp = get_tcp()
                        current = Position(tcp[0], tcp[1], tcp[2],
                                           tcp[3], tcp[4], tcp[5])
                        target = Position(home_pose[0], home_pose[1], home_pose[2],
                                          home_pose[3], home_pose[4], home_pose[5])
                        print ''
                        print '--- Returning to home pose ---'
                        print '  from: [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                            tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])
                        print '  to:   [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
                            home_pose[0], home_pose[1], home_pose[2],
                            home_pose[3], home_pose[4], home_pose[5])
                        rb.line(target)
                        move_history = []
                        print '--- Home pose reached ---'
                        show_pose()

                elif not move_history:
                    print 'Nothing to undo.'
                else:
                    valid_axes = set(['x', 'y', 'z', 'rx', 'ry', 'rz',
                                      'd1', 'd2', 'd3', 'd4', 'd5', 'd6'])

                    if len(args_list) == 0:
                        # undo: last 1
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
                        # undo all
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
                        # undo x / undo x ry rz
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
                            print 'Usage: undo / undo <N> / undo all / undo <axis...>'
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
                print 'Unknown: {}. (p/j/c/co/cc/scan/set/undo/go/gc/show/speed/q)'.format(cmd)

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
