#!/usr/bin/python
# -*- coding: utf-8 -*-
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

  --- 정렬 (Visual Servoing) ---
  detect            : 그리퍼 카메라로 큐브 검출, tvec 출력
  dset              : detect + 현재 tvec를 타겟으로 저장
  align             : XY 비주얼 서보잉 (타겟 tvec으로 수렴)
  alignz            : XY + Z 서보잉
  agrab             : align + Z 하강 + 그리퍼 닫기 (자동 잡기)

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

Step 0 — 축 매핑 캘리브레이션 (최초 1회)
──────────────────────────────────────────────────────────────
  그리퍼 카메라의 축과 로봇 TCP 축 간 매핑을 확인한다.
  이 매핑은 SERVO_CAM_X / SERVO_CAM_Y / SERVO_CAM_Z 상수로 설정.

  [로봇-서버]  python robot_calb.py
  [PC]         python Step2_in_capture_capture.py (또는 Step2_to)

  플로우:
    1. 큐브를 테이블에 놓고 그리퍼를 큐브 위에 대략 위치
    2. "detect"       → tvec [tx, ty, tz] 확인
    3. "p x,5"        → 로봇 X축 +5mm 이동
    4. "detect"       → tvec 변화 확인
       - tx가 변했으면: cam_x → robot x (부호도 확인)
       - ty가 변했으면: cam_y → robot x
    5. "undo"         → 원위치
    6. "p y,5" 후 "detect" → 같은 방식으로 Y축 확인
    7. 확인된 매핑을 robot_calb.py 상단 상수에 반영:
         SERVO_CAM_X = ('y', -1.0)
         SERVO_CAM_Y = ('x',  1.0)
         SERVO_CAM_Z = ('z', -1.0)

Step 1 — 카메라 내부 파라미터 (intrinsics)
──────────────────────────────────────────────────────────────
  [PC]
  python Step1_dump_all_intrinsics.py \
    --out_dir ./intrinsics \
    --gripper_serial <그리퍼_카메라_시리얼>

Step 1.5 — 그립 타겟 티칭 (dset)
──────────────────────────────────────────────────────────────
  큐브 돌출부의 마커 중점을 그리퍼가 정확히 잡는 위치를 가르친다.
  이후 agrab 명령으로 동일한 그립을 자동 재현할 수 있다.

  [로봇-서버]  python robot_calb.py
  [PC]         python Step2_in_capture_capture.py (또는 Step2_to)

  플로우:
    1. 큐브를 테이블에 놓음
    2. 수동으로 그리퍼를 큐브 위로 이동 (p x, p y, p z 등)
    3. 그리퍼 핑거팁이 돌출부(0.5mm)에 정확히 맞닿도록 위치 조정
       - 큐브 상면에서 2mm 아래로 들어가는 위치
       - 양쪽 핑거가 마커 중점 기준으로 대칭
    4. "gc"           → 그리퍼 닫기 (큐브 정확히 잡힘 확인)
    5. "dset"         → 현재 카메라-큐브 tvec을 타겟으로 저장
       *** Target tvec saved ***
       [0.0012, -0.0034, 0.1520] m
    6. "set"          → 현재 TCP도 홈 포즈로 저장 (선택)
    7. "go"           → 그리퍼 열기 (큐브 내려놓기)

  이제 다른 위치에서 "agrab"으로 동일한 그립을 자동 재현 가능.

Step 2a — 큐브 쥔 채로 촬영 (eye-in-hand)
──────────────────────────────────────────────────────────────
  큐브를 그리퍼로 쥔 상태에서 다양한 위치/자세로 이동하며 촬영.
  - 그리퍼 카메라: ChArUco 보드 + ArUco 큐브 검출
  - 고정 카메라: ArUco 큐브만 검출

  큐브를 잡을 때 "agrab"으로 마커 중점 정렬 후 잡으면
  Tool 4(큐브 중점)가 실제 마커 중점과 정확히 일치하여
  캘리브레이션 정확도가 향상된다.

  플로우:
    1. 큐브를 테이블에 놓음
    2. 그리퍼를 큐브 위로 대략 이동
    3. "agrab"        → 자동 정렬 + 하강 + 잡기
       (또는 수동: align → cc)
    4. 보드가 보이는 위치로 이동
    5. "c"            → 촬영 (그리퍼캠: 보드+큐브, 고정캠: 큐브)
    6. "p ry,15" / "p rz,30" 등으로 자세 변경 → "c" 반복
    7. 다른 위치로 이동 후 반복

  [로봇-서버]
  python robot_calb.py

  [PC]
  python Step2_in_capture_capture.py \
    --root_folder ./data/session_a \
    --intrinsics_dir ./intrinsics \
    --gripper_cam_idx 2 \
    --robot_ip 192.168.0.23 --robot_port 12348 \
    --show --also_detect_cube

Step 2b — 큐브 놓으면서 촬영 (hand-to-eye)
──────────────────────────────────────────────────────────────
  큐브를 보드 가장자리에 놓고 그리퍼를 올려서 촬영.
  - 그리퍼 카메라: ChArUco 보드 + ArUco 큐브 동시 검출
  - 고정 카메라: ArUco 큐브 검출

  agrab으로 잡은 큐브는 마커 중점에 정렬되어 있으므로
  놓을 때 기록되는 큐브 중점(Tool 4) 좌표가 정확하다.

  플로우:
    1. 큐브를 agrab으로 잡은 상태에서 보드 가장자리로 이동
    2. "scan"         → 자동: 놓기 → 촬영 → 집기
    3. "scan ry,15"   → 놓기 → 촬영 → 회전촬영 → 집기
    4. 다른 위치로 이동 후 반복

  [로봇-서버]
  python robot_calb.py

  [PC]
  python Step2_to_capture_capture.py \
    --root_folder ./data/session_b \
    --intrinsics_dir ./intrinsics \
    --robot_ip 192.168.0.23 --robot_port 12348 \
    --manual_robot --use_robot \
    --show --min_markers 1

Step 3a — Hand-Eye 캘리브레이션
──────────────────────────────────────────────────────────────
  Step 2a 데이터로 그리퍼↔카메라 변환행렬 계산.
  → 출력: T_gripper_cam.npy

  [PC]
  python Step3_in_calibration.py \
    --charuco_folder ./data/session_a \
    --intrinsics_dir ./intrinsics \
    --gripper_cam_idx 2

Step 3b — 멀티카메라 캘리브레이션
──────────────────────────────────────────────────────────────
  Step 2b 데이터로 고정카메라↔로봇베이스 변환행렬 계산.
  → 출력: T_gripper_cam.npy, T_base_C0~C3.npy, T_C0_C1.npy 등

  [PC]
  python Step3_to_calibration.py \
    --root_folder ./data/session_b \
    --intrinsics_dir ./intrinsics

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

# Gripper IO port
GRIPPER_IO_PORT = 48
GRIPPER_TIMEOUT_SEC = 5.0


# Cube grip parameters
CUBE_SIZE_MM = 30.0         # cube side length (mm)
CUBE_EDGE_MM = 2.0          # edge protrusion height (mm)
CUBE_GRIP_DEPTH_MM = 2.0    # fingertip enters 2mm below cube top
# TCP(fingertip) to cube center (along tool z): 15mm - 2mm = 13mm
CUBE_CENTER_OFFSET_Z = CUBE_SIZE_MM / 2.0 - CUBE_GRIP_DEPTH_MM
# co/cc lift height: grip depth(2) + edge(2) + margin(5) = 9mm above cube top
CUBE_LIFT_Z = CUBE_GRIP_DEPTH_MM + CUBE_EDGE_MM + 5.0

# Visual servoing parameters (for 'align' command)
# Camera-to-TCP axis mapping: (robot_tcp_axis, sign_multiplier)
# Calibrate once: run 'detect', then 'p x,5', then 'detect' again
#   Check which tvec component changed and by how much.
SERVO_CAM_X = ('y', -1.0)    # camera X -> robot TCP (axis, sign)
SERVO_CAM_Y = ('x',  1.0)    # camera Y -> robot TCP (axis, sign)
SERVO_CAM_Z = ('z', -1.0)    # camera Z (depth) -> robot TCP (axis, sign)
SERVO_GAIN = 0.7              # convergence gain (0~1)
SERVO_TOLERANCE_MM = 0.5      # XY alignment tolerance (mm)
SERVO_MAX_ITER = 15           # max servo iterations

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


def do_capture(conn, capture_count, cube_center_6dof=None, grip_target_tvec=None):
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
    if cube_center_6dof is not None:
        msg["cube_center_6dof"] = cube_center_6dof
    if grip_target_tvec is not None:
        msg["grip_target_tvec"] = grip_target_tvec

    send_json(conn, msg)

    resp = recv_json(conn)
    if resp is None:
        print 'Client disconnected!'
        return False

    status = resp.get('status', 'unknown') if isinstance(resp, dict) else 'unknown'
    print '*** Capture {} done (status={}) ***'.format(capture_count, status)
    print ''
    return True


def do_servo_xy(conn, target_tvec):
    """Visual servo: move TCP until camera XY aligns with target tvec.
    Returns (aligned, last_response)."""
    tgt_x, tgt_y, tgt_z = target_tvec[0], target_tvec[1], target_tvec[2]

    for i in range(SERVO_MAX_ITER):
        send_json(conn, {"command": "detect"})
        resp = recv_json(conn)

        if not resp or not resp.get("ok"):
            print '  [{}/{}] detection failed'.format(i + 1, SERVO_MAX_ITER)
            return False, resp

        tv = resp["tvec"]
        ex = (tv[0] - tgt_x) * 1000.0
        ey = (tv[1] - tgt_y) * 1000.0
        ez = (tv[2] - tgt_z) * 1000.0

        print '  [{}/{}] err: cx={:.2f} cy={:.2f} cz={:.1f} mm'.format(
            i + 1, SERVO_MAX_ITER, ex, ey, ez)

        if abs(ex) < SERVO_TOLERANCE_MM and abs(ey) < SERVO_TOLERANCE_MM:
            print '  XY aligned!'
            return True, resp

        ax_x, sign_x = SERVO_CAM_X
        ax_y, sign_y = SERVO_CAM_Y
        if abs(ex) >= SERVO_TOLERANCE_MM:
            move_tcp(ax_x, sign_x * ex * SERVO_GAIN)
        if abs(ey) >= SERVO_TOLERANCE_MM:
            move_tcp(ax_y, sign_y * ey * SERVO_GAIN)

    print '  Did not converge in {} iterations.'.format(SERVO_MAX_ITER)
    return False, None


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
        cube_center_6dof = None # cube center position when placed (grip corrected)
        holding_cube = True     # True: cube in gripper, False: cube placed
        home_pose = None        # saved TCP pose from 'set' command
        target_tvec = None      # [tx, ty, tz] meters, saved by 'dset'

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
        print '--- Align (visual servoing) ---'
        print '  detect            : Detect cube from gripper cam'
        print '  dset              : Detect + save as target'
        print '  align             : Servo XY to target'
        print '  alignz            : Servo XY + Z to target'
        print '  agrab             : Align + descend + grip'
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

                # --- CO: Place cube (read cube center via tool 4) ---
                cube_center_6dof = get_cube_center()

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
                ok = do_capture(conn, capture_count, cube_center_6dof, target_tvec)
                if not ok:
                    break
                capture_count += 1

                # --- Rotated captures ---
                for i, (ax, val) in enumerate(rotations):
                    print '--- Shot {}/{}: {} {} ---'.format(i + 2, n_shots, ax, val)
                    move_tcp(ax, val)
                    ok = do_capture(conn, capture_count, cube_center_6dof, target_tvec)
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
                    cube_center_6dof = get_cube_center()
                    print '  [Holding] cube center: [{:.1f}, {:.1f}, {:.1f}]'.format(
                        cube_center_6dof[0], cube_center_6dof[1], cube_center_6dof[2])
                ok = do_capture(conn, capture_count, cube_center_6dof, target_tvec)
                if not ok:
                    break
                capture_count += 1

            # ─── CO: Place cube (open gripper -> z up 22mm) ───
            elif cmd_lower == 'co':
                print ''
                print '--- CO: Place cube ---'

                # Read cube center via tool 4 BEFORE opening (cube is still held)
                cube_center_6dof = get_cube_center()
                print '  Cube center: [{:.1f}, {:.1f}, {:.1f}]'.format(
                    cube_center_6dof[0], cube_center_6dof[1], cube_center_6dof[2])

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

            # ─── DETECT: single cube detection from gripper cam ───
            elif cmd_lower == 'detect':
                send_json(conn, {"command": "detect"})
                resp = recv_json(conn)
                if resp and resp.get("ok"):
                    tv = resp["tvec"]
                    print ''
                    print '=== Cube Detected ==='
                    print '  tvec: [{:.4f}, {:.4f}, {:.4f}] m'.format(tv[0], tv[1], tv[2])
                    print '  tvec: [{:.2f}, {:.2f}, {:.2f}] mm'.format(
                        tv[0] * 1000, tv[1] * 1000, tv[2] * 1000)
                    print '  markers: {}'.format(resp.get("used_ids", []))
                    if target_tvec is not None:
                        ex = (tv[0] - target_tvec[0]) * 1000
                        ey = (tv[1] - target_tvec[1]) * 1000
                        ez = (tv[2] - target_tvec[2]) * 1000
                        print '  err vs target: cx={:.2f} cy={:.2f} cz={:.1f} mm'.format(
                            ex, ey, ez)
                    print ''
                else:
                    print 'Detection failed: {}'.format(resp)

            # ─── DSET: detect + save target tvec ───
            elif cmd_lower == 'dset':
                send_json(conn, {"command": "detect"})
                resp = recv_json(conn)
                if resp and resp.get("ok"):
                    tv = resp["tvec"]
                    target_tvec = tv
                    print ''
                    print '*** Target tvec saved ***'
                    print '  [{:.4f}, {:.4f}, {:.4f}] m'.format(tv[0], tv[1], tv[2])
                    print '  [{:.2f}, {:.2f}, {:.2f}] mm'.format(
                        tv[0] * 1000, tv[1] * 1000, tv[2] * 1000)
                    print ''
                else:
                    print 'Detection failed: {}'.format(resp)

            # ─── ALIGN: visual servoing XY (+ optional Z) ───
            elif cmd_lower.startswith('align'):
                if target_tvec is None:
                    print 'No target. Run "dset" first (grip cube correctly, then dset).'
                else:
                    do_z = 'z' in cmd_lower
                    print ''
                    print '=== ALIGN{} START ==='.format('+Z' if do_z else '')
                    print '  target: [{:.2f}, {:.2f}, {:.2f}] mm'.format(
                        target_tvec[0] * 1000, target_tvec[1] * 1000,
                        target_tvec[2] * 1000)

                    aligned, resp = do_servo_xy(conn, target_tvec)

                    if aligned and do_z and resp:
                        tv = resp["tvec"]
                        ez = (tv[2] - target_tvec[2]) * 1000.0
                        if abs(ez) > 1.0:
                            ax_z, sign_z = SERVO_CAM_Z
                            print '  Z correction: {:.1f}mm'.format(ez)
                            move_tcp(ax_z, sign_z * ez * SERVO_GAIN)

                    print '=== ALIGN END ==='
                    print ''

            # ─── AGRAB: align + descend + grip ───
            elif cmd_lower == 'agrab':
                if target_tvec is None:
                    print 'No target. Run "dset" first.'
                else:
                    print ''
                    print '=== AUTO GRAB ==='
                    print '  target: [{:.2f}, {:.2f}, {:.2f}] mm'.format(
                        target_tvec[0] * 1000, target_tvec[1] * 1000,
                        target_tvec[2] * 1000)

                    aligned, resp = do_servo_xy(conn, target_tvec)

                    if not aligned:
                        print '  XY align failed. Aborting.'
                        print '=== AUTO GRAB ABORTED ==='
                    else:
                        # Z descend to target
                        send_json(conn, {"command": "detect"})
                        resp2 = recv_json(conn)
                        if resp2 and resp2.get("ok"):
                            ez = (resp2["tvec"][2] - target_tvec[2]) * 1000.0
                            if abs(ez) > 1.0:
                                ax_z, sign_z = SERVO_CAM_Z
                                print '  Z descend: {:.1f}mm'.format(ez)
                                move_tcp(ax_z, sign_z * ez)

                        gripper_close()
                        holding_cube = True
                        print '=== AUTO GRAB DONE ==='
                    print ''

            else:
                print 'Unknown: {}'.format(cmd)

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
