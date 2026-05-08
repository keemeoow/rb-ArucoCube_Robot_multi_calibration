"""
캡처 세션 확장 서버 (Teach-Extend):
  PC로부터 기존 capture_waypoints.json을 받아 메모리에 적재한 뒤,
  - 기존 waypoint를 따라 이동(playall / playset / goto)하고
  - 사용자가 수동 이동 후 'c'로 새 포즈를 캡처해 append
  - 종료(q) 또는 'save' 시점에 머지된 전체 waypoints를 PC로 보내 저장
    (PC가 capture_waypoints.json.bak으로 백업한 뒤 덮어씀)

전제:
  - 큐브가 그리퍼에 잡혀 있는 상태로 시작.
  - PC측 Step2_capture.py가 manual robot 모드(`--use_robot --manual`)로 실행 중.
  - PC가 `request_waypoints` / `save_waypoints` 메시지를 처리할 수 있어야 함.

명령어:
  --- 조회 ---
  show              : 현재 TCP/관절 표시
  list              : 로드된 waypoints 요약
  status            : 현재 set 컨텍스트, 신규 추가 개수 등

  --- 수동 이동 ---
  p <축>,<값>       : TCP 상대 (z,50 / rx,15 ...)
  j <d#>,<값>       : 관절 상대 (d1,10 ...)
  gotop x,y,z[,rz,ry,rx] : TCP 절대 (line)
  gotoj d1..d6      : 관절 절대 (joint)
  speed <0-100>     : override 속도

  --- 그리퍼 ---
  go / gc           : 열기 / 닫기

  --- 재방문 (캡처 안 함, 이동만) ---
  playall [spd]     : 모든 set 순회 (set 진입 시 큐브 배치 → +Z 클리어런스 →
                      set 안의 capture_joints 들을 순서대로 방문 → 다음 set 전
                      재-그립 후 transit lift). 캡처는 PC로 보내지 않음.
  playset <S>       : set S로 진입 (필요 시 재-그립 후 이동, 큐브 release, +Z 클리어런스).
                      이 명령 후 'c'로 set S에 새 포즈를 추가할 수 있음.
  gotow <N>         : waypoint N(pose_index)의 capture_joints로 이동 (큐브 핸들링 없음).
  gotoplace <S>     : set S의 place_joints로 이동 (그리퍼 동작 없음).

  --- 신규 포즈 추가 ---
  setidx <S>        : 'c'가 어느 set에 귀속될지 set 컨텍스트 지정.
                      (playset 실행 시 자동으로 갱신됨)
  c                 : 현재 위치에서 캡처. PC가 이미지 저장 + 게이트 검사 후
                      성공 시 메모리 waypoints에 append (set_index = 현재 컨텍스트).
  undoc             : 마지막 신규 캡처 1건 취소(메모리만; PC 측 저장 이미지는 보존됨).

  --- 저장 / 종료 ---
  save              : 현재 메모리의 머지된 waypoints를 PC로 전송해 저장 (백업 후 덮어씀).
  q                 : 저장 후 종료. (신규 추가가 없으면 save 생략.)

CLI 옵션:
  --speed <N>       : 시작 시 override 속도 (기본 30).
  --local <path>    : PC fetch 대신 로컬 파일에서 waypoints 로드(테스트용).
                      이 경우 'save'는 PC 전송 대신 로컬 파일로 백업 후 덮어씀.
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
import os
import time
import socket
import json
import shutil

HOST = '0.0.0.0'
PORT = 12348

GRIPPER_IO_PORT = 48
GRIPPER_TIMEOUT_SEC = 5.0

CUBE_SIZE_MM = 30.0
CUBE_GRIP_DEPTH_MM = 2.0
CUBE_CENTER_OFFSET_Z = CUBE_SIZE_MM / 2.0 - CUBE_GRIP_DEPTH_MM

TOOL_GRIPPER_Z = 150.0
TOOL_CUBE_CENTER_Z = TOOL_GRIPPER_Z - CUBE_CENTER_OFFSET_Z

# 큐브 재-그립 시 +Z 위에서 line 접근 후 하강하여 닫기.
GRIP_APPROACH_Z_MM = 20.0
# set 사이 transit 시 큐브가 바닥을 끌지 않도록 +Z 들어올림.
TRANSIT_LIFT_MM = 30.0
# 큐브 release 후 캡처 영역으로 빠지기 위한 +Z.
CAPTURE_CLEARANCE_MM = 100.0

TCP_AXIS_MAP = {'x': 'dx', 'y': 'dy', 'z': 'dz', 'rz': 'drz', 'ry': 'dry', 'rx': 'drx'}
JOINT_AXIS_MAP = {'d1': 'dj1', 'd2': 'dj2', 'd3': 'dj3', 'd4': 'dj4', 'd5': 'dj5', 'd6': 'dj6'}


# ── Socket framing (newline-delimited JSON) ──

_RECV_BUF = {'data': b''}


def send_json(conn, obj):
    try:
        msg = json.dumps(obj)
        conn.sendall((msg + '\n').encode('utf-8'))
    except socket.error as e:
        print "Send error: {}".format(e)


def recv_json(conn):
    try:
        while b'\n' not in _RECV_BUF['data']:
            chunk = conn.recv(65536)
            if not chunk:
                if _RECV_BUF['data']:
                    line = _RECV_BUF['data']
                    _RECV_BUF['data'] = b''
                    try:
                        return json.loads(line.decode('utf-8').strip())
                    except Exception as e:
                        print "Recv parse error: {}".format(e)
                return None
            _RECV_BUF['data'] += chunk
        line, _, rest = _RECV_BUF['data'].partition(b'\n')
        _RECV_BUF['data'] = rest
        return json.loads(line.decode('utf-8').strip())
    except socket.error as e:
        print "Recv error: {}".format(e)
    except Exception as e:
        print "Recv parse error: {}".format(e)
    return None


# ── Robot helpers ──

def fmt6(v):
    return '[{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
        v[0], v[1], v[2], v[3], v[4], v[5])


def get_tcp():
    return rb.getpos().pos2list()[:6]


def get_joints():
    return rb.getjnt().jnt2list()[:6]


def get_cube_center():
    rb.changetool(4)
    tcp = rb.getpos().pos2list()[:6]
    rb.changetool(3)
    return tcp


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


def move_tcp(axis, value):
    if axis not in TCP_AXIS_MAP:
        print 'Invalid axis: {}. Use x,y,z,rz,ry,rx'.format(axis)
        return
    current = Position(*rb.getpos().pos2list()[:6])
    rb.line(current.offset(**{TCP_AXIS_MAP[axis]: value}))


def move_joint(axis, value):
    if axis not in JOINT_AXIS_MAP:
        print 'Invalid axis: {}. Use d1~d6'.format(axis)
        return
    current = Joint(*rb.getjnt().jnt2list()[:6])
    rb.move(current.offset(**{JOINT_AXIS_MAP[axis]: value}))


def line_dz(dz):
    cur = Position(*rb.getpos().pos2list()[:6])
    rb.line(cur.offset(dz=dz))


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


def approach_and_close(place_tcp):
    """+Z {GRIP_APPROACH_Z_MM}mm 위에서 line 접근 후 하강하여 닫기."""
    above = list(place_tcp[:6])
    above[2] += GRIP_APPROACH_Z_MM
    rb.line(Position(*above))
    time.sleep(0.2)
    rb.line(Position(*place_tcp[:6]))
    time.sleep(0.2)
    gripper_close()


# ── Waypoint indexing ──

def group_by_set(waypoints):
    """waypoints[]를 set_index 기준으로 first-appearance order로 그룹핑."""
    sets_order = []
    by_set = {}
    for wp in waypoints:
        sidx = wp.get('set_index')
        if sidx is None:
            continue
        if sidx not in by_set:
            by_set[sidx] = []
            sets_order.append(sidx)
        by_set[sidx].append(wp)
    return sets_order, by_set


def find_place_joints_for_set(waypoints, set_index):
    """해당 set의 첫 waypoint에서 place_joints를 가져온다."""
    for wp in waypoints:
        if wp.get('set_index') == set_index and 'place_joints' in wp:
            return wp['place_joints']
    return None


# ── Capture (PC와 통신) ──

def do_capture(conn, pose_index, set_cube_center=None, set_index=None,
               set_joints=None, set_tcp=None, place_joints=None):
    tcp = get_tcp()
    cube_tcp = get_cube_center()
    joints = get_joints()
    print ''
    print '*** CAPTURE pose_index={} (set={}) ***'.format(pose_index, set_index)
    print '  fingertip:   {}'.format(fmt6(tcp))
    print '  cube center: {}'.format(fmt6(cube_tcp))

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
        return None, None, None, None

    status = resp.get('status', 'unknown') if isinstance(resp, dict) else 'unknown'
    reason = resp.get('reason') if isinstance(resp, dict) else None
    if reason:
        print '*** capture done (status={}, reason={}) ***'.format(status, reason)
    else:
        print '*** capture done (status={}) ***'.format(status)
    return status, tcp, cube_tcp, joints


# ── Replay (no-capture navigation) ──

def navigate_set(rb, sidx, by_set, hold_state, override_speed):
    """현재 큐브 상태(hold_state)를 보고 set sidx 진입.

    hold_state: dict { 'held': bool, 'placed_at': int|None, 'last_place_tcp': list|None }
      - 'held' True 이면 그리퍼가 큐브를 들고 있다고 가정.
      - 'placed_at' 가 sidx와 같으면 이미 set에 있는 것으로 간주(아무 것도 안 함).
      - 'placed_at' 가 다른 set이면 그 set place_tcp에서 재-그립 후 이동.

    완료 후 hold_state를 갱신:
      - placed_at = sidx, held = False, last_place_tcp = 새 place_tcp
    """
    place_j = find_place_joints_for_set_in_dict(by_set, sidx)
    if place_j is None:
        print '[ERROR] set {} place_joints not found'.format(sidx)
        return False

    if hold_state.get('placed_at') == sidx and not hold_state.get('held'):
        print '[Replay] already at set {} (cube placed). nothing to do.'.format(sidx)
        return True

    # If cube currently placed at a different set, re-grip first.
    if hold_state.get('placed_at') is not None and not hold_state.get('held'):
        prev_tcp = hold_state.get('last_place_tcp')
        if prev_tcp is not None:
            print '[Replay] re-grip cube at previous set place pose'
            approach_and_close(prev_tcp)
        else:
            # fallback: move via prev set's place_joints
            prev_set = hold_state['placed_at']
            prev_j = find_place_joints_for_set_in_dict(by_set, prev_set)
            if prev_j is not None:
                rb.move(Joint(*prev_j[:6]))
                time.sleep(0.3)
                gripper_close()
        time.sleep(0.3)
        print '[Replay] +Z {:.0f}mm transit lift'.format(TRANSIT_LIFT_MM)
        line_dz(TRANSIT_LIFT_MM)
        time.sleep(0.3)
        hold_state['held'] = True
        hold_state['placed_at'] = None

    # Move to new set place_joints (joint move; cube descends to floor).
    print '[Replay] -> set {} place_joints'.format(sidx)
    rb.move(Joint(*place_j[:6]))
    time.sleep(0.4)
    place_tcp = get_tcp()

    # Release cube.
    print '[Replay] gripper OPEN (release cube)'
    gripper_open()
    time.sleep(0.3)

    # Lift +Z clearance for capture zone.
    print '[Replay] -> +Z {:.0f}mm clearance'.format(CAPTURE_CLEARANCE_MM)
    line_dz(CAPTURE_CLEARANCE_MM)
    time.sleep(0.3)

    hold_state['held'] = False
    hold_state['placed_at'] = sidx
    hold_state['last_place_tcp'] = place_tcp
    return True


def find_place_joints_for_set_in_dict(by_set, sidx):
    wps = by_set.get(sidx, [])
    for wp in wps:
        if 'place_joints' in wp:
            return wp['place_joints']
    return None


def replay_all(rb, conn, data, hold_state, override_speed):
    """모든 set/waypoint를 캡처 없이 순회. (검증용)"""
    waypoints = data.get('waypoints', [])
    if not waypoints:
        print '[ERROR] no waypoints'
        return
    sets_order, by_set = group_by_set(waypoints)
    if not sets_order:
        print '[ERROR] no set_index in waypoints'
        return

    print ''
    print '=========================================='
    print '  Replay (no capture)'
    print '  - sets:     {} ({})'.format(len(sets_order), sets_order)
    print '  - captures: {}'.format(len(waypoints))
    print '  - speed:    {}'.format(override_speed)
    print '=========================================='
    print ''
    print 'PRECONDITION: cube must currently be gripped.'
    raw_input('Press ENTER to start...')

    rb.override(override_speed)
    hold_state['held'] = True
    hold_state['placed_at'] = None
    hold_state['last_place_tcp'] = None

    # Initial transit lift so cube clears floor on first joint move.
    print '[Replay] +Z {:.0f}mm initial transit lift'.format(TRANSIT_LIFT_MM)
    try:
        line_dz(TRANSIT_LIFT_MM)
        time.sleep(0.3)
    except Exception as e:
        print '[WARN] initial lift failed: {} (continuing)'.format(e)

    for si, sidx in enumerate(sets_order):
        wps = by_set[sidx]
        print ''
        print '======== SET {}/{} (set_index={}, {} captures) ========'.format(
            si + 1, len(sets_order), sidx, len(wps))

        if not navigate_set(rb, sidx, by_set, hold_state, override_speed):
            print '[Replay] aborting'
            return

        # Visit each capture pose (no capture).
        for wi, wp in enumerate(wps):
            cap_j = wp.get('capture_joints')
            if cap_j is None:
                continue
            pose_idx = wp.get('pose_index', wi)
            print '  -> capture pose {}/{} (pose_index={})'.format(
                wi + 1, len(wps), pose_idx)
            try:
                rb.move(Joint(*cap_j[:6]))
                time.sleep(0.3)
            except Exception as e:
                print '  [WARN] move failed: {} (skip)'.format(e)

        # End of set: nothing to do here. Re-grip & transit happens when
        # navigate_set is called for the next set.

    print ''
    print '=========================================='
    print '  Replay done. Current set context = {}'.format(hold_state.get('placed_at'))
    print '  Use: c (capture+append) / playset <S> / setidx <S>'
    print '=========================================='


def replay_set(rb, conn, data, sidx, hold_state, override_speed):
    """단일 set으로 진입. 큐브 상태 자동 처리."""
    waypoints = data.get('waypoints', [])
    sets_order, by_set = group_by_set(waypoints)
    if sidx not in by_set:
        print '[ERROR] set {} not in waypoints (available: {})'.format(sidx, sets_order)
        return
    rb.override(override_speed)
    # Pre-lift if cube currently held but not yet on floor.
    if hold_state.get('held'):
        try:
            print '[Replay] +Z {:.0f}mm transit lift before set transition'.format(TRANSIT_LIFT_MM)
            line_dz(TRANSIT_LIFT_MM)
            time.sleep(0.3)
        except Exception as e:
            print '[WARN] lift failed: {}'.format(e)
    navigate_set(rb, sidx, by_set, hold_state, override_speed)


# ── PC sync ──

def request_waypoints_from_pc(conn, timeout_sec=10.0):
    print 'Requesting waypoints from PC...'
    send_json(conn, {"command": "request_waypoints"})
    conn.settimeout(timeout_sec)
    try:
        resp = recv_json(conn)
    except socket.timeout:
        print '[ERROR] PC did not respond within {}s'.format(timeout_sec)
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
    wp_data = resp.get('waypoints_data')
    if not isinstance(wp_data, dict):
        print '[ERROR] PC response missing waypoints_data'
        return None
    print '  received {} waypoints from PC'.format(len(wp_data.get('waypoints', [])))
    return wp_data


def send_save_to_pc(conn, data, timeout_sec=15.0):
    """PC에 머지된 waypoints를 보내 저장 요청. 성공 여부 반환."""
    print 'Sending merged waypoints to PC for save...'
    send_json(conn, {"command": "save_waypoints", "waypoints_data": data})
    conn.settimeout(timeout_sec)
    try:
        resp = recv_json(conn)
    except socket.timeout:
        print '[ERROR] PC did not respond to save_waypoints within {}s'.format(timeout_sec)
        return False
    finally:
        try:
            conn.settimeout(None)
        except Exception:
            pass
    if not isinstance(resp, dict) or resp.get('status') != 'ok':
        print '[ERROR] PC save failed: {}'.format(
            resp.get('reason', 'unknown') if isinstance(resp, dict) else resp)
        return False
    print '  PC saved {} waypoints.'.format(resp.get('n_waypoints', '?'))
    return True


def save_to_local(path, data):
    """로컬 파일 저장 (백업 후 덮어씀)."""
    if os.path.exists(path):
        bak = path + '.bak'
        try:
            shutil.copyfile(path, bak)
            print '  backup: {}'.format(bak)
        except Exception as e:
            print '[WARN] backup failed: {}'.format(e)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print '  saved: {} ({} waypoints)'.format(path, len(data.get('waypoints', [])))


# ── Main ──

def parse_argv():
    args = {'speed': 30, 'local': None}
    if '--speed' in sys.argv:
        i = sys.argv.index('--speed')
        if i + 1 < len(sys.argv):
            try:
                args['speed'] = int(sys.argv[i + 1])
            except ValueError:
                pass
    if '--local' in sys.argv:
        i = sys.argv.index('--local')
        if i + 1 < len(sys.argv):
            args['local'] = sys.argv[i + 1]
    return args


def main():
    cli = parse_argv()
    init_speed = cli['speed']
    local_path = cli['local']

    rbs = None
    s = None
    conn = None
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
        rb.override(init_speed)

        rb.settool(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rb.settool(2, 0.0, 35.0, 330.0, 0.0, 0.0, 0.0)
        rb.settool(3, 0.0, 0.0, TOOL_GRIPPER_Z, 0.0, 0.0, 0.0)
        rb.settool(4, 0.0, 0.0, TOOL_CUBE_CENTER_Z, 0.0, 0.0, 0.0)
        rb.changetool(3)
        rb.use_mt(True)

        # ── Connect (or load local) ──
        data = None
        if local_path:
            print 'Loading waypoints from local file: {}'.format(local_path)
            with open(local_path, 'r') as f:
                data = json.load(f)
            print '  loaded {} waypoints'.format(len(data.get('waypoints', [])))
            # 로컬 모드여도 capture/save 위해서 PC 연결은 시도. 실패해도 진행.
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print 'Server on port {} (waiting for PC; local mode)...'.format(PORT)
            conn, addr = s.accept()
            print 'Client: {}'.format(addr)
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            print 'Server on port {}. Waiting...'.format(PORT)
            conn, addr = s.accept()
            print 'Client: {}'.format(addr)
            data = request_waypoints_from_pc(conn)
            if data is None:
                print '[FATAL] could not load waypoints from PC. Exiting.'
                return

        waypoints = data.get('waypoints', [])
        sets_order, by_set = group_by_set(waypoints)
        existing_pose_max = max([wp.get('pose_index', -1) for wp in waypoints]) if waypoints else -1
        next_pose_index = existing_pose_max + 1
        n_existing = len(waypoints)

        # State
        current_set = sets_order[0] if sets_order else None
        hold_state = {'held': True, 'placed_at': None, 'last_place_tcp': None}
        new_count = 0

        print ''
        print '=========================================='
        print '  Teach-Extend Server'
        print '  - existing waypoints: {}'.format(n_existing)
        print '  - sets:               {}'.format(sets_order)
        print '  - next pose_index:    {}'.format(next_pose_index)
        print '  - speed:              {}'.format(init_speed)
        print '=========================================='
        print '  show / list / status / help'
        print '  p / j / gotop / gotoj / speed / go / gc'
        print '  playall [spd]  playset <S>  gotow <N>  gotoplace <S>'
        print '  setidx <S>     c            undoc'
        print '  save           q'
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

            if cl == 'q':
                break

            elif cl in ('help', '?'):
                print __doc__

            elif cl == 'show':
                show_pose()

            elif cl == 'list':
                print ''
                print '  pose_idx  set  capture_joints[d1,d2,d3,d4,d5,d6]'
                for wp in waypoints:
                    cj = wp.get('capture_joints', [0]*6)
                    print '  {:>8}  {:>3}  [{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}]'.format(
                        wp.get('pose_index'), wp.get('set_index'),
                        cj[0], cj[1], cj[2], cj[3], cj[4], cj[5])
                print '  total: {} waypoints (sets: {})'.format(len(waypoints), sets_order)
                print ''

            elif cl == 'status':
                print ''
                print '  current_set:   {}'.format(current_set)
                print '  hold_state:    held={}, placed_at={}'.format(
                    hold_state['held'], hold_state['placed_at'])
                print '  existing:      {}  new added: {}'.format(n_existing, new_count)
                print '  next pose_idx: {}'.format(next_pose_index)
                print ''

            elif cl.startswith('speed'):
                try:
                    spd = int(cmd.split()[1])
                    rb.override(spd)
                    print 'Speed: {}'.format(spd)
                except Exception:
                    print 'Usage: speed <0-100>'

            elif cl == 'go':
                gripper_open()
                hold_state['held'] = False

            elif cl == 'gc':
                gripper_close()
                hold_state['held'] = True

            elif cl.startswith('p '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis, value = parts[0].strip(), float(parts[1].strip())
                    move_tcp(axis, value)
                    show_pose()
                except Exception as e:
                    print 'Error: {}. Usage: p <axis>,<value>'.format(e)

            elif cl.startswith('j '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis, value = parts[0].strip(), float(parts[1].strip())
                    move_joint(axis, value)
                    show_pose()
                except Exception as e:
                    print 'Error: {}. Usage: j <axis>,<value>'.format(e)

            elif cl.startswith('gotop '):
                try:
                    vals = [float(v.strip()) for v in cmd[6:].strip().split(',')]
                    if len(vals) == 6:
                        rb.line(Position(*vals))
                    elif len(vals) == 3:
                        tcp = get_tcp()
                        rb.line(Position(vals[0], vals[1], vals[2], tcp[3], tcp[4], tcp[5]))
                    else:
                        print 'Usage: gotop x,y,z[,rz,ry,rx]'
                        continue
                    show_pose()
                except Exception as e:
                    print 'Error: {}'.format(e)

            elif cl.startswith('gotoj '):
                try:
                    vals = [float(v.strip()) for v in cmd[6:].strip().split(',')]
                    if len(vals) != 6:
                        print 'Usage: gotoj d1,d2,d3,d4,d5,d6'
                        continue
                    rb.move(Joint(*vals))
                    show_pose()
                except Exception as e:
                    print 'Error: {}'.format(e)

            elif cl.startswith('playall'):
                parts = cmd.split()
                spd = init_speed
                if len(parts) >= 2:
                    try:
                        spd = int(parts[1])
                    except ValueError:
                        pass
                replay_all(rb, conn, data, hold_state, spd)
                # After playall, current_set = last visited set.
                if hold_state.get('placed_at') is not None:
                    current_set = hold_state['placed_at']
                show_pose()

            elif cl.startswith('playset'):
                parts = cmd.split()
                if len(parts) < 2:
                    print 'Usage: playset <set_index>'
                    continue
                try:
                    sidx = int(parts[1])
                except ValueError:
                    print 'Usage: playset <set_index>'
                    continue
                replay_set(rb, conn, data, sidx, hold_state, init_speed)
                if hold_state.get('placed_at') == sidx:
                    current_set = sidx
                    print '[Info] current_set = {}'.format(current_set)
                show_pose()

            elif cl.startswith('gotow '):
                parts = cmd.split()
                if len(parts) < 2:
                    print 'Usage: gotow <pose_index>'
                    continue
                try:
                    n = int(parts[1])
                except ValueError:
                    print 'Usage: gotow <pose_index>'
                    continue
                target = None
                for wp in waypoints:
                    if wp.get('pose_index') == n:
                        target = wp
                        break
                if target is None or 'capture_joints' not in target:
                    print '[ERROR] waypoint {} not found'.format(n)
                    continue
                rb.move(Joint(*target['capture_joints'][:6]))
                show_pose()

            elif cl.startswith('gotoplace '):
                parts = cmd.split()
                if len(parts) < 2:
                    print 'Usage: gotoplace <set_index>'
                    continue
                try:
                    sidx = int(parts[1])
                except ValueError:
                    print 'Usage: gotoplace <set_index>'
                    continue
                pj = find_place_joints_for_set(waypoints, sidx)
                if pj is None:
                    print '[ERROR] set {} place_joints not found'.format(sidx)
                    continue
                rb.move(Joint(*pj[:6]))
                show_pose()

            elif cl.startswith('setidx'):
                parts = cmd.split()
                if len(parts) < 2:
                    print 'Usage: setidx <set_index>'
                    continue
                try:
                    sidx = int(parts[1])
                except ValueError:
                    print 'Usage: setidx <set_index>'
                    continue
                if sidx not in by_set:
                    print '[WARN] set {} not in existing waypoints; allowing anyway.'.format(sidx)
                current_set = sidx
                print 'current_set = {}'.format(current_set)

            elif cl == 'c':
                if current_set is None:
                    print '[ERROR] current_set is unset. Use playset <S> or setidx <S> first.'
                    continue
                if local_path:
                    # 로컬 모드: 캡처는 PC가 처리 못하므로 메모리에만 기록.
                    tcp = get_tcp()
                    cube_tcp = get_cube_center()
                    joints = get_joints()
                    print '*** [local-mode] capture (no PC roundtrip)'
                else:
                    pj = find_place_joints_for_set(waypoints, current_set)
                    sjoints = data.get('set_joints')
                    stcp = data.get('set_tcp')
                    scube = data.get('set_cube_center')
                    status, tcp, cube_tcp, joints = do_capture(
                        conn, next_pose_index,
                        set_cube_center=scube,
                        set_index=current_set,
                        set_joints=sjoints,
                        set_tcp=stcp,
                        place_joints=pj,
                    )
                    if status is None:
                        print '[FATAL] disconnected during capture'
                        break
                    if status != 'success':
                        print '  -> not appended (gate failed)'
                        continue
                wp = {
                    "pose_index": next_pose_index,
                    "capture_joints": joints,
                    "capture_tcp": tcp,
                    "cube_center_6dof": cube_tcp,
                    "set_index": current_set,
                }
                pj_for_save = find_place_joints_for_set(waypoints, current_set)
                if pj_for_save is not None:
                    wp["place_joints"] = pj_for_save
                waypoints.append(wp)
                # by_set도 갱신해야 다음 navigate_set이 새 entry를 찾을 수 있음.
                by_set.setdefault(current_set, []).append(wp)
                if current_set not in sets_order:
                    sets_order.append(current_set)
                next_pose_index += 1
                new_count += 1
                print '  -> appended pose_index={} (set={}). new total={}'.format(
                    wp['pose_index'], current_set, new_count)

            elif cl == 'undoc':
                if new_count == 0:
                    print 'No new captures to undo.'
                    continue
                # 가장 마지막에 추가된 신규 waypoint 제거
                last = waypoints.pop()
                # by_set에서도 제거
                bs = by_set.get(last.get('set_index'), [])
                if bs and bs[-1] is last:
                    bs.pop()
                next_pose_index -= 1
                new_count -= 1
                print 'Removed last new capture (pose_index={}). new total={}'.format(
                    last.get('pose_index'), new_count)

            elif cl == 'save':
                if new_count == 0:
                    print 'No new captures to save.'
                    continue
                data['waypoints'] = waypoints
                if local_path:
                    save_to_local(local_path, data)
                else:
                    if not send_save_to_pc(conn, data):
                        print '[WARN] PC save failed.'

            else:
                print 'Unknown: {}'.format(cmd)

        # ── Quit: auto-save if any new entries ──
        if new_count > 0:
            print ''
            print 'Saving {} new waypoints (total {})...'.format(new_count, len(waypoints))
            data['waypoints'] = waypoints
            if local_path:
                save_to_local(local_path, data)
            else:
                send_save_to_pc(conn, data)
        try:
            send_json(conn, {"command": "quit"})
        except Exception:
            pass

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
            if rb is not None:
                rb.exit(0)
                rb.close()
        except Exception:
            pass
        try:
            if rbs is not None:
                rbs.close()
        except Exception:
            pass
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
        try:
            if s is not None:
                s.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
