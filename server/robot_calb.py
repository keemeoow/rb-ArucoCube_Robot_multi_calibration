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
  set               : 현재 TCP + 관절값 + 큐브 중점(Tool 4) 저장
                      set_index (큐브 위치 #0, #1, ...) 자동 증가
                      촬영 시 TCP, 관절값, set 정보를 PC로 전송

  --- 그리퍼 ---
  go                : 그리퍼 열기
  gc                : 그리퍼 닫기

  --- 되돌리기 ---
  undo              : 마지막 이동 1회 되돌리기
  undo <N>          : 마지막 N회 되돌리기
  undo all          : 전체 되돌리기
  undo <axis...>    : 특정 축만 되돌리기 (undo x ry rz)
  undo set          : 어디서든 set 위치로 이동

  --- 자동화 ---
  start              : 멀티-set 자동 캡처 시작 (PC에서 waypoints 받음)
  start <path>       : 로컬 파일에서 waypoints 로드 (테스트용)
  start - <spd>      : PC에서 받기 + 속도 지정
  start <path> <spd> : 로컬 + 속도 지정
                      사전: 큐브가 그리퍼에 잡혀 있어야 함.
                      플로우: PC로부터 capture_waypoints.json 받음
                              -> +Z 30mm 초기 lift
                              -> set의 place_joints로 이동 -> 그리퍼 오픈
                              -> +Z 100mm 클리어런스 -> 각 capture_joints 촬영
                              -> 다음 set로 큐브 이동 (re-grip + +Z 30mm transit lift)
                              -> 반복.

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

  플로우 (큐브 위치별 멀티 캘리브레이션):
    1. 큐브를 바닥에 놓기
    2. "set"          -> 큐브 위치 #N 저장 (TCP, 관절값, 큐브 중점)
    3. 그리퍼 카메라 위치를 다양하게 이동
    4. "c"            -> 촬영 (TCP, 관절값, set_index가 PC로 전송)
    5. 3~4 반복 (같은 큐브 위치에서 여러 각도)
    6. 큐브를 새 위치로 옮기고 2~5 반복
    * PC에 capture_waypoints.json + meta.json 동시 저장

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
import sys
import time
import socket
import json

HOST = '0.0.0.0'
PORT = 12348

GRIPPER_IO_PORT = 48
GRIPPER_TIMEOUT_SEC = 5.0

CUBE_SIZE_MM = 30.0
CUBE_GRIP_DEPTH_MM = 2.0
CUBE_CENTER_OFFSET_Z = CUBE_SIZE_MM / 2.0 - CUBE_GRIP_DEPTH_MM

TOOL_GRIPPER_Z = 150.0
TOOL_CUBE_CENTER_Z = TOOL_GRIPPER_Z - CUBE_CENTER_OFFSET_Z

TCP_AXIS_MAP = {'x': 'dx', 'y': 'dy', 'z': 'dz', 'rz': 'drz', 'ry': 'dry', 'rx': 'drx'}
JOINT_AXIS_MAP = {'d1': 'dj1', 'd2': 'dj2', 'd3': 'dj3', 'd4': 'dj4', 'd5': 'dj5', 'd6': 'dj6'}
VALID_AXES = set(list(TCP_AXIS_MAP.keys()) + list(JOINT_AXIS_MAP.keys()))


# ── Socket ──

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


# ── Robot helpers ──

def fmt6(v):
    return '[{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
        v[0], v[1], v[2], v[3], v[4], v[5])


def get_tcp():
    return rb.getpos().pos2list()[:6]


def get_cube_center():
    rb.changetool(4)
    tcp = rb.getpos().pos2list()[:6]
    rb.changetool(3)
    return tcp


def get_joints():
    return rb.getjnt().jnt2list()[:6]


def show_pose():
    tcp = get_tcp()
    jnt = get_joints()
    print ''
    print '=== TCP Pose ==='
    print '  x={:.3f}  y={:.3f}  z={:.3f}'.format(tcp[0], tcp[1], tcp[2])
    print '  rz={:.3f}  ry={:.3f}  rx={:.3f}'.format(tcp[3], tcp[4], tcp[5])
    print '=== Joints ==='
    print '  d1={:.3f}  d2={:.3f}  d3={:.3f}'.format(jnt[0], jnt[1], jnt[2])
    print '  d4={:.3f}  d5={:.3f}  d6={:.3f}'.format(jnt[3], jnt[4], jnt[5])
    print ''
    return tcp


def move_tcp(axis, value):
    if axis not in TCP_AXIS_MAP:
        print 'Invalid axis: {}. Use x,y,z,rz,ry,rx'.format(axis)
        return
    current = Position(*rb.getpos().pos2list()[:6])
    rb.line(current.offset(**{TCP_AXIS_MAP[axis]: value}))
    print 'TCP {} += {} done'.format(axis, value)


def move_joint(axis, value):
    if axis not in JOINT_AXIS_MAP:
        print 'Invalid axis: {}. Use d1~d6'.format(axis)
        return
    current = Joint(*rb.getjnt().jnt2list()[:6])
    rb.move(current.offset(**{JOINT_AXIS_MAP[axis]: value}))
    print 'Joint {} += {} done'.format(axis, value)


def undo_one(entry):
    mtype, maxis, mvalue = entry
    print '  {} {},{} -> {}'.format(mtype, maxis, mvalue, -mvalue)
    if mtype == 'p':
        move_tcp(maxis, -mvalue)
    else:
        move_joint(maxis, -mvalue)


# ── Gripper ──

def check_gripper():
    return [din(GRIPPER_IO_PORT + i) for i in [3, 2, 1, 0]]


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


# ── Capture ──

def do_capture(conn, pose_index, set_cube_center=None, set_index=None,
               set_joints=None, set_tcp=None, place_joints=None):
    """Returns (status, tcp, cube_tcp) or (None, None, None) on disconnect."""
    tcp = get_tcp()
    cube_tcp = get_cube_center()
    joints = get_joints()
    print ''
    print '*** CAPTURE {} ***'.format(pose_index)
    print '  fingertip:    {}'.format(fmt6(tcp))
    print '  cube center:  {}'.format(fmt6(cube_tcp))

    msg = {
        "command": "capture",
        "capture_pose_6dof": tcp,
        "cube_center_pose_6dof": cube_tcp,
        "robot_joints_6dof": joints,
        "pose_index": pose_index,
    }
    if set_cube_center is not None:
        msg["set_cube_center_6dof"] = set_cube_center
    if set_index is not None:
        msg["set_index"] = set_index
    if set_joints is not None:
        msg["set_joints"] = set_joints
    if set_tcp is not None:
        msg["set_tcp"] = set_tcp
    if place_joints is not None:
        msg["place_joints"] = place_joints

    send_json(conn, msg)
    resp = recv_json(conn)
    if resp is None:
        print 'Client disconnected!'
        return None, None, None

    status = resp.get('status', 'unknown') if isinstance(resp, dict) else 'unknown'
    reason = resp.get('reason') if isinstance(resp, dict) else None
    if reason:
        print '*** Capture {} done (status={}, reason={}) ***'.format(pose_index, status, reason)
    else:
        print '*** Capture {} done (status={}) ***'.format(pose_index, status)
    return status, tcp, cube_tcp


# ── Auto capture ──

def request_waypoints_from_pc(conn, timeout_sec=10.0):
    """Request capture_waypoints.json content from the PC over the socket.

    Returns the parsed dict on success, or None on failure / timeout.
    """
    print 'Requesting waypoints from PC...'
    send_json(conn, {"command": "request_waypoints"})
    conn.settimeout(timeout_sec)
    try:
        resp = recv_json(conn)
    except socket.timeout:
        print '[ERROR] PC did not respond within {}s'.format(timeout_sec)
        conn.settimeout(None)
        return None
    finally:
        try:
            conn.settimeout(None)
        except Exception:
            pass
    if not isinstance(resp, dict):
        print '[ERROR] invalid response from PC'
        return None
    if resp.get('status') != 'ok':
        print '[ERROR] PC reported error: {}'.format(resp.get('reason', 'unknown'))
        return None
    data = resp.get('waypoints_data')
    if not isinstance(data, dict):
        print '[ERROR] PC response missing waypoints_data'
        return None
    n_wps = len(data.get('waypoints', []))
    print '  received {} waypoints from PC'.format(n_wps)
    return data


def run_auto_capture(rb, conn, waypoint_file=None, speed=30):
    """Run auto capture. If waypoint_file is None or empty, request waypoints
    from PC over the socket. Otherwise, load from local filesystem (legacy)."""
    if not waypoint_file:
        data = request_waypoints_from_pc(conn)
        if data is None:
            return
    else:
        with open(waypoint_file, 'r') as f:
            data = json.load(f)

    # Multi-set joint-based: waypoints[] has per-waypoint set_index (5+ sets)
    waypoints = data.get('waypoints', [])
    if waypoints and any('set_index' in wp for wp in waypoints):
        _run_auto_multiset(rb, conn, data, speed)
    elif data.get('format_version') == 2 and 'placements' in data:
        _run_auto_v2(rb, conn, data, speed)
    else:
        _run_auto_v1(rb, conn, data, speed)


def _run_auto_multiset(rb, conn, data, speed,
                        z_clearance_mm=100.0,
                        z_transit_lift_mm=30.0):
    """Multi-set joint-based auto capture (start-command flow).

    Per set in capture_waypoints.json (waypoints[].set_index grouping):
      1. Approach: move via place_joints + z_transit_lift_mm (line down) so the
         cube lowers gently to the floor instead of arriving via direct joint
         interpolation (avoids floor contact during transit).
      2. Open gripper -> cube released on the floor.
      3. Move up by z_clearance_mm in +Z direction (line motion in TCP frame).
      4. For each waypoint in this set: move to capture_joints, capture.
      5. If a next set exists: return to place_joints, close gripper, line-lift
         +z_transit_lift_mm before the next set's joint transit (cube clears
         the floor every time it is moved between sets).
    """
    waypoints = data.get('waypoints', [])
    if not waypoints:
        print '[ERROR] no waypoints'
        send_json(conn, {"command": "quit"})
        return

    # Group by set_index, preserving first-appearance order.
    sets_order = []
    by_set = {}
    for wp in waypoints:
        sidx = wp.get('set_index')
        if sidx is None:
            print '[ERROR] waypoint pose_index={} missing set_index'.format(wp.get('pose_index'))
            send_json(conn, {"command": "quit"})
            return
        if 'place_joints' not in wp or 'capture_joints' not in wp:
            print '[ERROR] waypoint pose_index={} missing place_joints/capture_joints'.format(wp.get('pose_index'))
            send_json(conn, {"command": "quit"})
            return
        if sidx not in by_set:
            by_set[sidx] = []
            sets_order.append(sidx)
        by_set[sidx].append(wp)

    total_caps = len(waypoints)
    n_sets = len(sets_order)
    print ''
    print '=========================================='
    print '  Multi-Set Auto Capture'
    print '  - sets:     {} ({})'.format(n_sets, sets_order)
    print '  - captures: {}'.format(total_caps)
    print '  - speed:    {}'.format(speed)
    print '  - +Z capture clearance: {}mm'.format(z_clearance_mm)
    print '  - +Z transit lift:      {}mm'.format(z_transit_lift_mm)
    print '=========================================='
    print ''
    print 'PRECONDITION: cube must be gripped before starting.'
    print 'Robot will move to set {}\'s place_joints first, then release the cube.'.format(sets_order[0])
    raw_input('Press ENTER to confirm cube is gripped and start...')

    rb.override(speed)
    success = 0
    skipped = 0

    # Pre-loop: cube is on the floor with the gripper around it (user just placed
    # it there). Lift +z_transit_lift_mm before the first joint transit to make
    # sure the cube clears the floor on the way to set 0's place_joints.
    print '[Auto] +Z {:.0f}mm initial transit lift (cube clears floor)'.format(z_transit_lift_mm)
    try:
        cur = Position(*rb.getpos().pos2list()[:6])
        rb.line(cur.offset(dz=z_transit_lift_mm))
    except Exception as e:
        print '[WARN] initial transit lift failed: {} (continuing)'.format(e)
    time.sleep(0.3)

    for si, sidx in enumerate(sets_order):
        wps = by_set[sidx]
        place_j = wps[0]['place_joints']

        print ''
        print '======== SET {}/{} (set_index={}, {} captures) ========'.format(
            si + 1, n_sets, sidx, len(wps))

        # Step 1: move to place_joints (cube placement position). Robot is
        # already lifted +z_transit_lift_mm from the previous set transit (or
        # is at the start position holding the cube), so the joint motion to
        # place_joints brings the cube down to the floor naturally.
        print '[Auto] -> set {} place_joints'.format(sidx)
        rb.move(Joint(*place_j[:6]))
        time.sleep(0.5)

        # Step 2: open gripper -> cube released.
        print '[Auto] gripper OPEN (release cube on floor)'
        gripper_open()
        time.sleep(0.3)

        # Step 3: clearance up in +Z (capture clearance — bigger than transit).
        print '[Auto] -> +Z {:.0f}mm clearance'.format(z_clearance_mm)
        try:
            cur = Position(*rb.getpos().pos2list()[:6])
            rb.line(cur.offset(dz=z_clearance_mm))
        except Exception as e:
            print '[WARN] +Z clearance failed: {} (continuing)'.format(e)
        time.sleep(0.5)

        # Step 4: visit each capture pose.
        for wi, wp in enumerate(wps):
            cap_j = wp['capture_joints']
            pose_idx = wp.get('pose_index', wi)
            print '  -- capture {}/{} pose_index={} --'.format(
                wi + 1, len(wps), pose_idx)
            try:
                rb.move(Joint(*cap_j[:6]))
            except Exception as e:
                print '  [WARN] move failed: {}. Skipping.'.format(e)
                skipped += 1
                continue
            time.sleep(0.5)

            status, _, _ = do_capture(
                conn, pose_idx,
                set_cube_center=data.get('set_cube_center'),
                set_index=sidx,
                set_joints=place_j,
                set_tcp=None,
                place_joints=place_j,
            )
            if status is None:
                print '[Auto] disconnected, stopping.'
                return
            if status == 'success':
                success += 1
                print '  [Auto] -> OK'
            else:
                skipped += 1
                print '  [Auto] -> SKIPPED'

        # Step 5: if more sets remain, re-grip the cube and lift before transit.
        if si < n_sets - 1:
            print '[Auto] returning to place_joints to re-grip cube...'
            rb.move(Joint(*place_j[:6]))
            time.sleep(0.3)
            gripper_close()
            time.sleep(0.3)
            # Lift +Z transit_lift_mm so the cube clears the floor during the
            # joint transit to the next set's place_joints.
            print '[Auto] +Z {:.0f}mm transit lift (cube clears floor)'.format(z_transit_lift_mm)
            try:
                cur = Position(*rb.getpos().pos2list()[:6])
                rb.line(cur.offset(dz=z_transit_lift_mm))
            except Exception as e:
                print '[WARN] transit lift failed: {} (continuing)'.format(e)
            time.sleep(0.3)

    # Final state: gripper open at last set's place_joints (cube on floor).
    send_json(conn, {"command": "quit"})
    print ''
    print '=========================================='
    print '  Multi-Set Auto Complete'
    print '  - success: {}/{}'.format(success, total_caps)
    print '  - skipped: {}'.format(skipped)
    print '=========================================='


def _run_auto_v1(rb, conn, data, speed):
    """Legacy: flat waypoints list, joint-based captures, 1 capture per pick-place cycle."""
    set_joints = data.get('set_joints')
    set_cube_center = data.get('set_cube_center')
    wps = data.get('waypoints', [])

    if set_joints is None:
        print '[ERROR] set_joints not found!'
        send_json(conn, {"command": "quit"})
        return

    missing = [i for i, wp in enumerate(wps)
               if 'place_joints' not in wp or 'capture_joints' not in wp]
    if missing:
        print '[ERROR] Missing joints in waypoints: {}'.format(missing)
        send_json(conn, {"command": "quit"})
        return

    print ''
    print '=========================================='
    print '  Auto Capture v1: {} waypoints, speed={}'.format(len(wps), speed)
    print '=========================================='

    rb.override(speed)
    print '[Auto] Moving to SET...'
    rb.move(Joint(*set_joints[:6]))
    print '[Auto] At SET. Ensure cube is gripped!'
    raw_input('Press ENTER to start...')

    success_count = 0

    for i, wp in enumerate(wps):
        place_j = wp['place_joints']
        capture_j = wp['capture_joints']
        print ''
        print '======== Waypoint {}/{} ========'.format(i + 1, len(wps))

        rb.move(Joint(*place_j[:6]))
        time.sleep(0.3)
        gripper_open()
        time.sleep(0.3)

        rb.move(Joint(*capture_j[:6]))
        time.sleep(0.5)

        status, _, _ = do_capture(conn, i, set_cube_center)
        if status is None:
            break
        if status == 'success':
            success_count += 1
            print '[Auto] -> OK'
        else:
            print '[Auto] -> SKIPPED'

        rb.move(Joint(*place_j[:6]))
        time.sleep(0.3)
        gripper_close()
        time.sleep(0.3)
        rb.move(Joint(*set_joints[:6]))

    send_json(conn, {"command": "quit"})
    print ''
    print '  Auto Complete v1: {}/{} captured'.format(success_count, len(wps))


def _run_auto_v2(rb, conn, data, speed):
    """v2: per-placement multi-capture. Place once, capture N TCP poses, re-grip, next placement."""
    set_joints = data.get('set_joints')
    set_tcp = data.get('set_tcp')
    placements = data.get('placements', [])

    if set_joints is None:
        print '[ERROR] set_joints not found!'
        send_json(conn, {"command": "quit"})
        return

    for i, p in enumerate(placements):
        if 'place_joints' not in p:
            print '[ERROR] placement {} missing place_joints'.format(i)
            send_json(conn, {"command": "quit"})
            return
        for j, cap in enumerate(p.get('captures', [])):
            if 'capture_tcp' not in cap:
                print '[ERROR] placement {} capture {} missing capture_tcp'.format(i, j)
                send_json(conn, {"command": "quit"})
                return

    total_caps = sum(len(p.get('captures', [])) for p in placements)
    print ''
    print '=========================================='
    print '  Auto Capture v2: {} placements, {} captures, speed={}'.format(
        len(placements), total_caps, speed)
    print '=========================================='

    rb.override(speed)
    print '[Auto] Moving to SET...'
    rb.move(Joint(*set_joints[:6]))
    print '[Auto] At SET. Ensure cube is gripped!'
    raw_input('Press ENTER to start...')

    success_count = 0
    skipped_count = 0
    disconnected = False

    for pi, placement in enumerate(placements):
        if disconnected:
            break

        place_j = placement['place_joints']
        captures = placement.get('captures', [])
        set_idx = placement.get('set_index')
        set_cube = placement.get('set_cube_center')

        print ''
        print '======== Placement {}/{} (set_index={}, {} captures) ========'.format(
            pi + 1, len(placements), set_idx, len(captures))

        rb.move(Joint(*place_j[:6]))
        time.sleep(0.3)
        gripper_open()
        time.sleep(0.3)

        for ci, cap in enumerate(captures):
            tcp = cap['capture_tcp']
            pose_idx = cap.get('pose_index', ci)
            print '  -- capture {}/{} pose_index={} --'.format(
                ci + 1, len(captures), pose_idx)

            try:
                rb.line(Position(*tcp[:6]))
            except Exception as e:
                print '  [WARN] move failed: {}. Skipping.'.format(e)
                skipped_count += 1
                continue
            time.sleep(0.5)

            status, _, _ = do_capture(
                conn, pose_idx, set_cube,
                set_index=set_idx,
                set_joints=set_joints,
                set_tcp=set_tcp,
                place_joints=place_j)

            if status is None:
                print '  [Auto] Disconnected, stopping.'
                disconnected = True
                break
            elif status == 'success':
                success_count += 1
                print '  [Auto] -> OK'
            else:
                skipped_count += 1
                print '  [Auto] -> SKIPPED'

        if disconnected:
            break

        # Return to place pose, re-grip cube, return to SET
        print '  [Auto] Re-gripping cube and returning to SET...'
        rb.move(Joint(*place_j[:6]))
        time.sleep(0.3)
        gripper_close()
        time.sleep(0.3)
        rb.move(Joint(*set_joints[:6]))

    if not disconnected:
        send_json(conn, {"command": "quit"})
    print ''
    print '  Auto Complete v2: {}/{} captured ({} skipped)'.format(
        success_count, total_caps, skipped_count)


# ── Main ──

def main():
    try:
        rbs = RobSys()
        rbs.open()

        global rb
        rb = i611Robot()
        Base()
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
        rb.changetool(3)
        rb.use_mt(True)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print "Server on port {}. Waiting...".format(PORT)

        conn, addr = s.accept()
        print "Client: {}".format(addr)

        # Auto mode
        if '--auto' in sys.argv:
            idx = sys.argv.index('--auto')
            auto_file = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'capture_waypoints.json'
            auto_speed = 30
            if '--speed' in sys.argv:
                sidx = sys.argv.index('--speed')
                if sidx + 1 < len(sys.argv):
                    auto_speed = int(sys.argv[sidx + 1])
            try:
                run_auto_capture(rb, conn, auto_file, auto_speed)
            finally:
                try:
                    conn.close()
                    s.close()
                except Exception:
                    pass
            return

        # State
        capture_count = 0
        set_index = -1
        move_history = []
        home_pose = None
        home_joints = None
        set_cube_center = None
        last_place_joints = None
        waypoints = []

        print ''
        print '=========================================='
        print '  p <a>,<v> / j <a>,<v> : move'
        print '  goto x,y,z[,rz,ry,rx] : abs move'
        print '  show / speed <0-100>'
        print '  c: capture  set: save TCP+cube'
        print '  go: grip open  gc: grip close'
        print '  undo [N|all|<axes>|set]  q: quit'
        print '  start                 -> auto capture (PC sends waypoints)'
        print '  start <path> [speed]  -> auto capture (local file)'
        print '    (cube must be gripped before start)'
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

            cl = cmd.lower()

            # Quit
            if cl == 'q':
                send_json(conn, {"command": "quit"})
                break

            # Start: multi-set auto capture.
            #   "start"         -> request waypoints from PC over socket
            #   "start <path>"  -> load local file (legacy/testing)
            #   "start <path> <speed>" or "start - <speed>" supported
            elif cl.startswith('start'):
                parts = cmd.split(None, 2)
                wp_file = None
                spd = 30
                if len(parts) >= 2 and parts[1] != '-':
                    wp_file = parts[1]
                if len(parts) >= 3:
                    try:
                        spd = int(parts[2])
                    except ValueError:
                        print '[ERROR] invalid speed: {}'.format(parts[2])
                        continue
                try:
                    run_auto_capture(rb, conn, wp_file, spd)
                except IOError as e:
                    print '[ERROR] cannot read {}: {}'.format(wp_file, e)
                    continue
                break

            # Show
            elif cl == 'show':
                show_pose()
                if home_pose is not None:
                    print '  [Set #{}] TCP:  {}'.format(set_index, fmt6(home_pose))
                if set_cube_center is not None:
                    print '  [Set #{}] Cube: [{:.1f}, {:.1f}, {:.1f}]'.format(
                        set_index, set_cube_center[0], set_cube_center[1], set_cube_center[2])

            # Speed
            elif cl.startswith('speed'):
                try:
                    spd = int(cmd.split()[1])
                    rb.override(spd)
                    print 'Speed: {}'.format(spd)
                except Exception:
                    print 'Usage: speed <0-100>'

            # Set
            elif cl == 'set':
                set_index += 1
                home_pose = get_tcp()
                home_joints = get_joints()
                set_cube_center = get_cube_center()
                move_history = []
                print ''
                print '*** Set #{} saved ***'.format(set_index)
                print '  TCP:    {}'.format(fmt6(home_pose))
                print '  Joints: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(
                    home_joints[0], home_joints[1], home_joints[2],
                    home_joints[3], home_joints[4], home_joints[5])
                print '  Cube:   [{:.1f}, {:.1f}, {:.1f}] (offset={:.0f}mm)'.format(
                    set_cube_center[0], set_cube_center[1], set_cube_center[2],
                    CUBE_CENTER_OFFSET_Z)

            # Gripper
            elif cl == 'go':
                last_place_joints = get_joints()
                gripper_open()

            elif cl == 'gc':
                gripper_close()

            # Capture
            elif cl == 'c':
                status, tcp, cube_tcp = do_capture(
                    conn, capture_count, set_cube_center,
                    set_index if set_index >= 0 else None,
                    set_joints=home_joints, set_tcp=home_pose,
                    place_joints=last_place_joints)
                if status is None:
                    break
                wp = {
                    "pose_index": capture_count,
                    "capture_joints": get_joints(),
                    "capture_tcp": tcp,
                    "cube_center_6dof": cube_tcp,
                }
                if last_place_joints is not None:
                    wp["place_joints"] = last_place_joints
                else:
                    print '  [WARN] go not called before capture'
                waypoints.append(wp)
                capture_count += 1

            # Undo
            elif cl.startswith('undo'):
                args = cl.split()[1:]

                if args == ['set']:
                    if home_pose is None:
                        print 'No set saved.'
                    else:
                        target = Position(home_pose[0], home_pose[1], 0.0,
                                          home_pose[3], home_pose[4], home_pose[5])
                        rb.line(target)
                        move_history = []
                        show_pose()

                elif not move_history:
                    print 'Nothing to undo.'

                else:
                    if not args:
                        undo_one(move_history.pop())

                    elif args[0] == 'all':
                        while move_history:
                            undo_one(move_history.pop())

                    elif args[0] in VALID_AXES:
                        axis_set = set(a for a in args if a in VALID_AXES)
                        indices = [i for i, h in enumerate(move_history) if h[1] in axis_set]
                        if not indices:
                            print 'No moves on [{}]'.format(','.join(sorted(axis_set)))
                        else:
                            for idx in reversed(indices):
                                undo_one(move_history.pop(idx))
                    else:
                        try:
                            count = min(int(args[0]), len(move_history))
                        except ValueError:
                            print 'Usage: undo [N|all|<axes>|set]'
                            continue
                        for _ in range(count):
                            undo_one(move_history.pop())

                    show_pose()

            # Goto
            elif cl.startswith('goto '):
                try:
                    vals = [float(v.strip()) for v in cmd[5:].strip().split(',')]
                    if len(vals) == 6:
                        rb.line(Position(*vals))
                    elif len(vals) == 3:
                        tcp = get_tcp()
                        rb.line(Position(vals[0], vals[1], vals[2], tcp[3], tcp[4], tcp[5]))
                    else:
                        print 'Usage: goto x,y,z[,rz,ry,rx]'
                        continue
                    show_pose()
                except Exception as e:
                    print 'Error: {}'.format(e)

            # TCP move
            elif cl.startswith('p '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis, value = parts[0].strip(), float(parts[1].strip())
                    move_tcp(axis, value)
                    move_history.append(('p', axis, value))
                    show_pose()
                except Exception as e:
                    print 'Error: {}. Usage: p <axis>,<value>'.format(e)

            # Joint move
            elif cl.startswith('j '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis, value = parts[0].strip(), float(parts[1].strip())
                    move_joint(axis, value)
                    move_history.append(('j', axis, value))
                    show_pose()
                except Exception as e:
                    print 'Error: {}. Usage: j <axis>,<value>'.format(e)

            else:
                print 'Unknown: {}'.format(cmd)

        # Save waypoints
        if waypoints:
            save_data = {
                "set_joints": home_joints,
                "set_tcp": home_pose,
                "set_cube_center": set_cube_center,
                "waypoints": waypoints,
            }
            with open('capture_waypoints.json', 'w') as f:
                json.dump(save_data, f, indent=2)
            print '\nWaypoints saved: {} poses'.format(len(waypoints))

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
