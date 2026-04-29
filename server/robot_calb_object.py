"""
로봇 물체 촬영 서버 (Teach-and-Capture for Object Pose):
  수동 조작으로 로봇을 이동/회전하면서 물체 RGB-D 이벤트를 수집하는 서버.
  기본 가정은 object-fixed viewpoint sweep이다.

명령어:
  --- 이동 ---
  p <축>,<값>       : TCP 상대 이동 (예: "p z,50", "p rz,15")
  j <축>,<값>       : 관절 상대 이동 (예: "j d1,10")
  goto x,y,z,rz,ry,rx : TCP 절대 좌표로 이동
  show              : 현재 TCP 포즈 및 관절 값 표시
  speed <0-100>     : 속도 설정 (클수록 빠름)

  --- 촬영 & 중지 ---
  c                 : 현재 위치에서 촬영
  x                 : 촬영 + auto-replay 중지 (soft stop) 
  X(shift + x)      : 촬영 + auto-replay 중지 (hard abort) 

  --- 설정 ---
  set               : 현재 TCP + 관절값을 object station 기준점으로 저장
                      set_index (station #0, #1, ...) 자동 증가
                      촬영 시 TCP, 관절값, station 정보를 PC로 전송
  mode <auto|placed|held|object_fixed>
                    : 촬영 모드 지정. object_fixed면 물체는 station에 고정되고
                      로봇은 gripper camera viewpoint만 바꾼다.

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

Step 2 -- 물체 놓으면서 촬영
--------------------------------------------------------------
  수동으로 object station 기준 자세를 "set" 저장.
  이후 viewpoint를 바꾸며 촬영 -> "undo set" 복귀 반복.

  - 그리퍼 카메라: 근접 object view
  - 고정 카메라: 전역 anchor view

  [로봇-서버]
  python server/robot_calb_object\ .py

  [PC]
  python Object_6Dpose_estimation/Obj_Step1_capture_object.py \
    --save_dir ./data/object_capture \
    --intrinsics_dir ./intrinsics \
    --calib_dir ./data/session/calib_out\(2\) \
    --object_glb_dir ./Object_6Dpose_estimation/reference_glb \
    --use_robot --manual_robot \
    --robot_ip 192.168.0.23 --robot_port 12348 \
    --show --save_depth

  플로우 (object station별 멀티뷰 촬영):
    1. 물체를 놓을 station 기준 위치에서 "set"
    2. 권장: "mode object_fixed" 지정
    3. 촬영 자세로 이동
    4. "c" -> 촬영 (TCP, joints, station, place, capture_mode, gripper_state 전송)
    5. 반복
    * PC에 meta.json 동시 저장

Step 3 -- 물체 6D pose 추정
--------------------------------------------------------------
  [PC]
  python Object_6Dpose_estimation/Obj_Step2_pose_per_object.py \
    --capture_dir ./data/object_capture \
    --intrinsics_dir ./intrinsics \
    --calib_dir ./data/session/calib_out\(2\) \
    --object_glb_dir ./Object_6Dpose_estimation/reference_glb

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
import select
import threading

HOST = '0.0.0.0'
PORT = 12348

GRIPPER_IO_PORT = 48
GRIPPER_TIMEOUT_SEC = 5.0

TOOL_GRIPPER_Z = 150.0
TOOL_OBJECT_REF_Z = TOOL_GRIPPER_Z

STATIC_WINDOW_SEC = 0.4
STATIC_SAMPLES = 5
STATIC_MAX_JOINT_DELTA_DEG = 0.10
STATIC_MAX_TCP_TRANSLATION_MM = 1.0
STATIC_MAX_TCP_ROTATION_DEG = 0.20

VALID_CAPTURE_MODES = ('auto', 'placed', 'held', 'object_fixed')

# 자동 촬영(replay) 중 사용자가 'stop'/'x'/'abort'/'xx' 를 입력하면 set.
# run_auto_capture 와 inline_replay_current_station 의 viewpoint 루프가
# 이걸 체크해서 다음 viewpoint 시작 전에 빠져나간다 (cooperative).
# 콘솔에서 입력하면 현재 진행 중인 rb.move() 자체도 rb.stop()/rb.abort() 로
# 즉시 인터럽트 된다 (immediate).
stop_flag = threading.Event()
abort_flag = threading.Event()


def _drain_stdin():
    """stdin 에 남은 입력을 깨끗하게 비운다 (auto 종료 후 잔류 키 방지)."""
    try:
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                break
            sys.stdin.readline()
    except Exception:
        pass


def _stdin_stop_listener(rb_ref):
    """auto-replay 중 별도 thread 에서 stdin 을 폴링.

    인식 명령:
      stop / x   : rb.stop()  (soft, 감속 정지) + stop_flag.set()
      abort / xx : rb.abort() (hard, 즉시 정지) + stop_flag.set() + abort_flag.set()

    rb.stop() / rb.abort() 는 현재 진행 중인 rb.move() 를 즉시 인터럽트.
    """
    try:
        while not stop_flag.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not r:
                continue
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break
            cmd = line.strip().lower()
            if cmd in ('stop', 'x'):
                stop_flag.set()
                print ''
                print '[STOP] received -- rb.stop() (soft, decelerating)...'
                try:
                    rb_ref.stop()
                except Exception as exc:
                    print '[STOP] rb.stop() failed: {}'.format(exc)
                break
            elif cmd in ('abort', 'xx'):
                stop_flag.set()
                abort_flag.set()
                print ''
                print '[ABORT] received -- rb.abort() (hard, motion interrupted)...'
                try:
                    rb_ref.abort()
                except Exception as exc:
                    print '[ABORT] rb.abort() failed: {}'.format(exc)
                break
            elif cmd:
                print '[INFO] auto-replay 중에는 stop/x (soft) 또는 abort/xx (hard) 만 인식 (입력: {!r})'.format(cmd)
    except Exception as exc:
        print '[stop_listener] {}'.format(exc)


def _safe_move(rb_ref, target):
    """rb.move() 또는 rb.line() wrapper. stop_flag set 상태이면 호출 자체를 skip
    하고, move 도중 rb.stop()/rb.abort() 로 인터럽트되어 예외가 떠도 흡수한다.

    Returns: True 정상 완료 / False 인터럽트되거나 skip."""
    if stop_flag.is_set():
        return False
    try:
        rb_ref.move(target)
        return True
    except Exception as exc:
        if stop_flag.is_set():
            print '[move] interrupted by stop/abort: {}'.format(exc)
            return False
        # 비정지 예외는 다시 던짐
        raise

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


def _max_abs_diff(values_a, values_b):
    return max(abs(float(a) - float(b)) for a, b in zip(values_a, values_b))


def _tcp_translation_delta_mm(tcp_a, tcp_b):
    return ((float(tcp_a[0]) - float(tcp_b[0])) ** 2 +
            (float(tcp_a[1]) - float(tcp_b[1])) ** 2 +
            (float(tcp_a[2]) - float(tcp_b[2])) ** 2) ** 0.5


def _tcp_rotation_delta_deg(tcp_a, tcp_b):
    return max(abs(float(tcp_a[i]) - float(tcp_b[i])) for i in [3, 4, 5])


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


def get_gripper_state():
    io_bits = check_gripper()
    if io_bits == ['0', '1', '0', '0']:
        return 'open'
    if io_bits == ['0', '0', '0', '1']:
        return 'closed'
    return 'unknown'


def resolve_capture_mode(gripper_state, mode_override):
    if mode_override in ('placed', 'held', 'object_fixed'):
        return mode_override
    if gripper_state == 'closed':
        return 'held'
    return 'placed'


def normalize_pose6(values, label, allow_none=False):
    if values is None:
        if allow_none:
            return None
        raise ValueError('{} is required'.format(label))
    if not isinstance(values, (list, tuple)):
        raise ValueError('{} must be a list/tuple'.format(label))
    if len(values) < 6:
        raise ValueError('{} must have 6 values'.format(label))
    out = [float(v) for v in values[:6]]
    if len(values) != 6:
        print '[WARN] {} has {} values; using first 6.'.format(label, len(values))
    return out


def resolve_station_index(payload):
    if not isinstance(payload, dict):
        return None
    for key in ('station_id', 'station_index', 'set_index'):
        raw = payload.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            raise ValueError('Invalid {}: {}'.format(key, raw))
    return None


def normalize_waypoint_name(raw_name, pose_index):
    if raw_name is None:
        return 'viewpoint_{:02d}'.format(int(pose_index))
    name = str(raw_name).strip()
    if not name:
        return 'viewpoint_{:02d}'.format(int(pose_index))
    return name


def build_station_waypoint_payload(station_id, set_joints, set_tcp, capture_modes_seen, waypoints):
    capture_modes_seen = sorted(set(str(mode) for mode in capture_modes_seen if mode))
    if capture_modes_seen == ['object_fixed']:
        capture_mode = 'object_fixed'
        session_mode = 'object_fixed_viewpoint_sweep'
        file_prefix = 'object_fixed_waypoints'
    else:
        capture_mode = capture_modes_seen[0] if len(capture_modes_seen) == 1 else 'object_capture'
        session_mode = 'object_capture'
        file_prefix = 'object_capture_waypoints'
    payload = {
        "session_mode": session_mode,
        "capture_mode": capture_mode,
        "capture_modes_seen": capture_modes_seen,
        "station_id": int(station_id),
        "station_index": int(station_id),
        "set_index": int(station_id),
        "set_joints": normalize_pose6(set_joints, 'set_joints'),
        "set_tcp": normalize_pose6(set_tcp, 'set_tcp', allow_none=True),
        "waypoints": waypoints,
    }
    filename = '{}_station{:02d}.json'.format(file_prefix, int(station_id))
    return payload, filename


def save_station_waypoint_files(station_records):
    saved = []
    for station_id in sorted(station_records.keys()):
        rec = station_records[station_id]
        if not rec.get('waypoints'):
            continue
        payload, filename = build_station_waypoint_payload(
            station_id=station_id,
            set_joints=rec.get('set_joints'),
            set_tcp=rec.get('set_tcp'),
            capture_modes_seen=rec.get('capture_modes_seen', []),
            waypoints=rec.get('waypoints', []),
        )
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
        saved.append(filename)
    return saved


def load_object_fixed_waypoints(waypoint_file):
    with open(waypoint_file, 'r') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError('Waypoint file must be a JSON object: {}'.format(waypoint_file))
    if 'set_cube_center' in data:
        raise ValueError(
            'Cube calibration waypoint JSON is not supported for object-fixed replay: {}'.format(
                waypoint_file
            )
        )

    set_joints = normalize_pose6(data.get('set_joints'), 'set_joints')
    set_tcp = normalize_pose6(data.get('set_tcp'), 'set_tcp', allow_none=True)
    wps_raw = data.get('waypoints')
    if not isinstance(wps_raw, list) or not wps_raw:
        raise ValueError('No waypoints found in {}'.format(waypoint_file))

    capture_mode = str(data.get('capture_mode', 'object_fixed')).strip() or 'object_fixed'
    if capture_mode == 'auto':
        capture_mode = 'object_fixed'
    elif capture_mode in ('placed', 'held'):
        raise ValueError(
            '{} declares capture_mode={}; object-fixed auto replay only accepts object_fixed waypoint files.'.format(
                waypoint_file, capture_mode
            )
        )
    if capture_mode not in VALID_CAPTURE_MODES:
        raise ValueError('Unsupported capture_mode={} in {}'.format(capture_mode, waypoint_file))
    if capture_mode != 'object_fixed':
        raise ValueError(
            '{} must use capture_mode=object_fixed for station-fixed viewpoint replay.'.format(
                waypoint_file
            )
        )

    station_id = resolve_station_index(data)
    seen_station_ids = set()
    normalized = []
    for idx, wp in enumerate(wps_raw):
        if not isinstance(wp, dict):
            raise ValueError('waypoints[{}] must be a JSON object'.format(idx))
        if 'cube_center_6dof' in wp:
            raise ValueError(
                'waypoints[{}] looks like cube calibration data; object-fixed replay requires station-local viewpoints.'.format(
                    idx
                )
            )
        pose_index = int(wp.get('pose_index', idx))
        waypoint_name = normalize_waypoint_name(wp.get('name'), pose_index)
        capture_joints = normalize_pose6(
            wp.get('capture_joints'),
            'waypoints[{}].capture_joints'.format(idx),
        )
        capture_tcp = normalize_pose6(
            wp.get('capture_tcp'),
            'waypoints[{}].capture_tcp'.format(idx),
            allow_none=True,
        )
        place_joints = normalize_pose6(
            wp.get('place_joints'),
            'waypoints[{}].place_joints'.format(idx),
            allow_none=True,
        )
        place_tcp = normalize_pose6(
            wp.get('place_tcp'),
            'waypoints[{}].place_tcp'.format(idx),
            allow_none=True,
        )
        wp_station = resolve_station_index(wp)
        if wp_station is not None:
            seen_station_ids.add(int(wp_station))
        normalized.append({
            "pose_index": pose_index,
            "name": waypoint_name,
            "capture_joints": capture_joints,
            "capture_tcp": capture_tcp,
            "place_joints": place_joints,
            "place_tcp": place_tcp,
            "station_index": wp_station,
        })

    if station_id is None:
        if len(seen_station_ids) > 1:
            raise ValueError(
                'Mixed station_index/set_index values found in {}; split each station into a separate object-fixed file.'.format(
                    waypoint_file
                )
            )
        station_id = list(seen_station_ids)[0] if seen_station_ids else 0

    if len(seen_station_ids) > 1:
        raise ValueError(
            'Mixed station_index/set_index values found in {}; object-fixed auto replay expects one station per file.'.format(
                waypoint_file
            )
        )
    if seen_station_ids and int(list(seen_station_ids)[0]) != int(station_id):
        raise ValueError(
            'Top-level station_id={} does not match waypoint station={}'.format(
                station_id, list(seen_station_ids)[0]
            )
        )

    for wp in normalized:
        wp['station_index'] = int(station_id)
        wp['set_index'] = int(station_id)
        if wp['place_joints'] is None:
            wp['place_joints'] = list(set_joints)
        if wp['place_tcp'] is None and set_tcp is not None:
            wp['place_tcp'] = list(set_tcp)

    return {
        "session_mode": "object_fixed_viewpoint_sweep",
        "capture_mode": capture_mode,
        "station_id": int(station_id),
        "set_joints": set_joints,
        "set_tcp": set_tcp,
        "waypoints": normalized,
    }


def sample_robot_static(window_sec=STATIC_WINDOW_SEC,
                        sample_count=STATIC_SAMPLES,
                        joint_thr_deg=STATIC_MAX_JOINT_DELTA_DEG,
                        tcp_thr_mm=STATIC_MAX_TCP_TRANSLATION_MM,
                        rot_thr_deg=STATIC_MAX_TCP_ROTATION_DEG):
    joints_hist = []
    tcp_hist = []
    dt = float(window_sec) / float(max(sample_count - 1, 1))
    for idx in range(max(int(sample_count), 2)):
        joints_hist.append(get_joints())
        tcp_hist.append(get_tcp())
        if idx + 1 < max(int(sample_count), 2):
            time.sleep(dt)

    ref_joints = joints_hist[0]
    ref_tcp = tcp_hist[0]
    max_joint_delta_deg = max(_max_abs_diff(jv, ref_joints) for jv in joints_hist)
    max_tcp_translation_mm = max(_tcp_translation_delta_mm(tv, ref_tcp) for tv in tcp_hist)
    max_tcp_rotation_deg = max(_tcp_rotation_delta_deg(tv, ref_tcp) for tv in tcp_hist)
    static_ok = (
        max_joint_delta_deg <= float(joint_thr_deg) and
        max_tcp_translation_mm <= float(tcp_thr_mm) and
        max_tcp_rotation_deg <= float(rot_thr_deg)
    )
    return {
        'ok': bool(static_ok),
        'sample_count': int(len(joints_hist)),
        'window_sec': float(window_sec),
        'max_joint_delta_deg': float(max_joint_delta_deg),
        'max_tcp_translation_mm': float(max_tcp_translation_mm),
        'max_tcp_rotation_deg': float(max_tcp_rotation_deg),
        'joint_threshold_deg': float(joint_thr_deg),
        'tcp_translation_threshold_mm': float(tcp_thr_mm),
        'tcp_rotation_threshold_deg': float(rot_thr_deg),
        'last_tcp': tcp_hist[-1],
        'last_joints': joints_hist[-1],
    }


# ── Capture ──

def do_capture(conn, pose_index, set_index=None, set_joints=None,
               set_tcp=None, place_joints=None, place_tcp=None,
               capture_mode_override='auto', station_index=None,
               viewpoint_name=None):
    """Returns (status, tcp) or (None, None) on disconnect."""
    static_info = sample_robot_static()
    tcp = static_info['last_tcp']
    joints = static_info['last_joints']
    gripper_state = get_gripper_state()
    capture_mode = resolve_capture_mode(gripper_state, capture_mode_override)
    station_id = station_index if station_index is not None else set_index
    print ''
    print '*** CAPTURE {} ***'.format(pose_index)
    if viewpoint_name:
        print '  viewpoint:    {}'.format(viewpoint_name)
    if station_id is not None:
        print '  station:      {}'.format(station_id)
    print '  fingertip:    {}'.format(fmt6(tcp))
    print '  joints:       [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(
        joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])
    print '  gripper:      {} | capture_mode={} | static={}'.format(
        gripper_state, capture_mode, 'PASS' if static_info['ok'] else 'FAIL')

    msg = {
        "command": "capture",
        "capture_pose_6dof": tcp,
        "robot_joints_6dof": joints,
        "pose_index": pose_index,
        "station_index": station_id,
        "capture_mode": capture_mode,
        "session_mode": 'object_fixed_viewpoint_sweep' if capture_mode == 'object_fixed' else 'object_capture',
        "gripper_state": gripper_state,
        "robot_static_ok": bool(static_info['ok']),
        "robot_static_metrics": static_info,
        "robot_moving": bool(not static_info['ok']),
    }
    if station_id is not None:
        msg["station_id"] = station_id
        msg["set_index"] = station_id
    if set_joints is not None:
        msg["set_joints"] = set_joints
    if set_tcp is not None:
        msg["set_tcp"] = set_tcp
    if place_joints is not None:
        msg["place_joints"] = place_joints
    if place_tcp is not None:
        msg["place_pose_6dof"] = place_tcp
    if viewpoint_name:
        msg["viewpoint_name"] = viewpoint_name

    send_json(conn, msg)
    resp = recv_json(conn)
    if resp is None:
        print 'Client disconnected!'
        return None, None

    # PC 클라가 'captured' 대신 stop/abort 를 보내면 auto-replay 루프 abort.
    # (preview 창에서 x = soft, X = hard 키 누른 경우)
    if isinstance(resp, dict):
        action = resp.get('action')
        if action == 'stop':
            stop_flag.set()
            print '*** Capture {} -- client requested STOP (soft) ***'.format(pose_index)
            try:
                rb.stop()
            except Exception as exc:
                print '[STOP] rb.stop() failed: {}'.format(exc)
            return 'skipped', tcp
        if action == 'abort':
            stop_flag.set()
            abort_flag.set()
            print '*** Capture {} -- client requested ABORT (hard) ***'.format(pose_index)
            try:
                rb.abort()
            except Exception as exc:
                print '[ABORT] rb.abort() failed: {}'.format(exc)
            return 'skipped', tcp

    status = resp.get('status', 'unknown') if isinstance(resp, dict) else 'unknown'
    reason = resp.get('reason') if isinstance(resp, dict) else None
    if reason:
        print '*** Capture {} done (status={}, reason={}) ***'.format(pose_index, status, reason)
    else:
        print '*** Capture {} done (status={}) ***'.format(pose_index, status)
    return status, tcp


# ── Auto capture ──

def run_auto_capture(rb, conn, waypoint_file, speed=30):
    try:
        plan = load_object_fixed_waypoints(waypoint_file)
    except Exception as exc:
        print '[ERROR] {}'.format(exc)
        send_json(conn, {"command": "quit"})
        return

    set_joints = plan['set_joints']
    set_tcp = plan.get('set_tcp')
    station_id = plan['station_id']
    capture_mode = plan['capture_mode']
    wps = plan['waypoints']

    print ''
    print '=========================================='
    print '  Auto Capture: {} viewpoints, speed={}'.format(len(wps), speed)
    print '  mode={} station_id={}'.format(capture_mode, station_id)
    print '=========================================='
    print '  Object stays fixed on the station.'
    print '  Robot kinematics are used only for live gripper-camera extrinsics.'

    rb.override(speed)
    print '[Auto] Moving to SET...'
    if not _safe_move(rb, Joint(*set_joints[:6])):
        print '[Auto] aborted before sweep started.'
        return 0, True
    print '[Auto] At SET. Ensure the object is fixed and the gripper will not touch it.'
    while True:
        try:
            line = raw_input("Type 'start' (or 'quit') to begin viewpoint sweep: ").strip().lower()
        except EOFError:
            line = 'quit'
        if line == 'start':
            break
        if line in ('quit', 'q', 'exit'):
            print '[Auto] aborted before sweep started.'
            return 0, True
        if line:
            print "  type 'start' or 'quit' (got: {!r})".format(line)

    success_count = 0
    aborted = False

    for i, wp in enumerate(wps):
        if stop_flag.is_set():
            print '[Auto] stop_flag set -- aborting before viewpoint {}/{}'.format(i + 1, len(wps))
            aborted = True
            break
        place_j = wp.get('place_joints') or list(set_joints)
        capture_j = wp['capture_joints']
        viewpoint_name = wp.get('name')
        pose_index = int(wp.get('pose_index', i))
        print ''
        print '======== Viewpoint {}/{}: {} ========'.format(
            i + 1, len(wps), viewpoint_name
        )

        if not _safe_move(rb, Joint(*place_j[:6])):
            aborted = True; break
        time.sleep(0.3)

        if not _safe_move(rb, Joint(*capture_j[:6])):
            aborted = True; break
        time.sleep(0.5)

        status, _ = do_capture(
            conn,
            pose_index,
            set_index=station_id,
            set_joints=set_joints,
            set_tcp=set_tcp,
            place_joints=place_j,
            place_tcp=wp.get('place_tcp'),
            capture_mode_override=capture_mode,
            station_index=station_id,
            viewpoint_name=viewpoint_name,
        )
        if status is None:
            break
        if status == 'success':
            success_count += 1
            print '[Auto] -> OK'
        else:
            print '[Auto] -> SKIPPED'

        if not _safe_move(rb, Joint(*place_j[:6])):
            aborted = True; break
        time.sleep(0.2)
        if not _safe_move(rb, Joint(*set_joints[:6])):
            aborted = True; break
        time.sleep(0.2)

    if aborted:
        print '  Auto Aborted: {}/{} captured at station {}'.format(
            success_count, len(wps), station_id
        )
    else:
        send_json(conn, {"command": "quit"})
        print ''
        print '  Auto Complete: {}/{} captured at station {}'.format(
            success_count, len(wps), station_id
        )
    return success_count, aborted


def inline_replay_current_station(rb, conn, station_id, station_rec, speed=30):
    """현재 station_records 에 누적된 waypoints 를 in-memory 로 즉시 replay.
    (run_auto_capture 와 거의 동일하지만 파일 로드 대신 dict 를 받음.)
    stop_flag 가 set 되면 다음 viewpoint 시작 전에 빠져나간다."""
    set_joints = station_rec.get('set_joints')
    set_tcp = station_rec.get('set_tcp')
    wps = station_rec.get('waypoints') or []
    if not set_joints:
        print '[InlineAuto] station #{} has no set_joints'.format(station_id)
        return 0, True
    if not wps:
        print '[InlineAuto] station #{} has no waypoints. Capture some with `c` first.'.format(station_id)
        return 0, True

    capture_modes = station_rec.get('capture_modes_seen') or []
    if capture_modes and len(set(capture_modes)) == 1:
        capture_mode = capture_modes[0]
    else:
        capture_mode = 'object_fixed'

    print ''
    print '=========================================='
    print '  Inline Replay: station #{}, {} viewpoints, speed={}'.format(
        station_id, len(wps), speed)
    print '  mode={}'.format(capture_mode)
    print '=========================================='
    print '  Type "stop" or "x" + ENTER to abort between viewpoints.'

    rb.override(speed)
    print '[InlineAuto] Moving to SET...'
    if not _safe_move(rb, Joint(*set_joints[:6])):
        print '[InlineAuto] aborted before sweep started.'
        return 0, True
    time.sleep(0.3)

    success_count = 0
    aborted = False
    for i, wp in enumerate(wps):
        if stop_flag.is_set():
            print '[InlineAuto] stop_flag set -- aborting before viewpoint {}/{}'.format(i + 1, len(wps))
            aborted = True
            break
        place_j = wp.get('place_joints') or list(set_joints)
        capture_j = wp.get('capture_joints')
        if capture_j is None:
            print '[InlineAuto] viewpoint {} has no capture_joints, skipping'.format(i)
            continue
        viewpoint_name = wp.get('name') or normalize_waypoint_name(None, i)
        pose_index = int(wp.get('pose_index', i))
        print ''
        print '======== Replay {}/{}: {} ========'.format(
            i + 1, len(wps), viewpoint_name)

        if not _safe_move(rb, Joint(*place_j[:6])):
            aborted = True; break
        time.sleep(0.3)
        if not _safe_move(rb, Joint(*capture_j[:6])):
            aborted = True; break
        time.sleep(0.5)

        status, _ = do_capture(
            conn, pose_index,
            set_index=station_id,
            set_joints=set_joints, set_tcp=set_tcp,
            place_joints=place_j, place_tcp=wp.get('place_tcp'),
            capture_mode_override=capture_mode,
            station_index=station_id,
            viewpoint_name=viewpoint_name,
        )
        if status is None:
            break
        if status == 'success':
            success_count += 1
            print '[InlineAuto] -> OK'
        else:
            print '[InlineAuto] -> SKIPPED'

        if not _safe_move(rb, Joint(*place_j[:6])):
            aborted = True; break
        time.sleep(0.2)
        if not _safe_move(rb, Joint(*set_joints[:6])):
            aborted = True; break
        time.sleep(0.2)

    print ''
    if aborted:
        print '  Inline Replay Aborted: {}/{} captured'.format(success_count, len(wps))
    else:
        print '  Inline Replay Complete: {}/{} captured'.format(success_count, len(wps))
    return success_count, aborted


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
        rb.settool(4, 0.0, 0.0, TOOL_OBJECT_REF_Z, 0.0, 0.0, 0.0)
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
        capture_mode_override = 'object_fixed'
        last_place_joints = None
        last_place_tcp = None
        station_records = {}
        unassigned_waypoints = []

        print ''
        print '=========================================='
        print '  p <a>,<v> / j <a>,<v> : move'
        print '  goto x,y,z[,rz,ry,rx] : abs move'
        print '  show / speed <0-100>'
        print '  c: capture  set: save station TCP/joints'
        print '  mode <auto|placed|held|object_fixed>: capture mode override'
        print '  go: grip open  gc: grip close'
        print '  undo [N|all|<axes>|set]'
        print '  auto [file] [speed] : replay current station OR a JSON file'
        print '  stop / x   : rb.stop()  (soft, decelerating)'
        print '  abort / xx : rb.abort() (hard, motion interrupted)'
        print '  (during auto-replay, type stop/x/abort/xx + ENTER)'
        print '  q: quit'
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

            # Show
            elif cl == 'show':
                show_pose()
                if home_pose is not None:
                    print '  [Station #{}] TCP:  {}'.format(set_index, fmt6(home_pose))
                print '  Capture mode: {}'.format(capture_mode_override)
                if capture_mode_override == 'object_fixed':
                    print '  Semantics: object fixed on station, gripper viewpoint sweep only'
                print '  Gripper state: {}'.format(get_gripper_state())

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
                move_history = []
                last_place_joints = None
                last_place_tcp = None
                station_records[set_index] = {
                    "set_joints": list(home_joints),
                    "set_tcp": list(home_pose),
                    "capture_modes_seen": [],
                    "waypoints": [],
                }
                print ''
                print '*** Station #{} saved ***'.format(set_index)
                print '  TCP:    {}'.format(fmt6(home_pose))
                print '  Joints: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(
                    home_joints[0], home_joints[1], home_joints[2],
                    home_joints[3], home_joints[4], home_joints[5])
                print '  Mode:   {}'.format(capture_mode_override)
                if capture_mode_override == 'object_fixed':
                    print '  Object-fixed semantics enabled for this station.'

            elif cl.startswith('mode '):
                new_mode = cl.split(' ', 1)[1].strip()
                if new_mode not in VALID_CAPTURE_MODES:
                    print 'Usage: mode <auto|placed|held|object_fixed>'
                else:
                    capture_mode_override = new_mode
                    print 'Capture mode override: {}'.format(capture_mode_override)

            # Gripper
            elif cl == 'go':
                if capture_mode_override == 'object_fixed':
                    print '[WARN] object_fixed mode usually does not require gripper open/close.'
                last_place_joints = get_joints()
                last_place_tcp = get_tcp()
                gripper_open()

            elif cl == 'gc':
                if capture_mode_override == 'object_fixed':
                    print '[WARN] object_fixed mode usually does not require gripper open/close.'
                gripper_close()

            # Capture
            elif cl == 'c':
                if capture_mode_override == 'object_fixed' and set_index < 0:
                    print '[WARN] Save the station first with `set` before object_fixed capture.'
                    continue
                status, tcp = do_capture(
                    conn, capture_count,
                    set_index if set_index >= 0 else None,
                    set_joints=home_joints, set_tcp=home_pose,
                    place_joints=last_place_joints,
                    place_tcp=last_place_tcp,
                    capture_mode_override=capture_mode_override,
                    station_index=set_index if set_index >= 0 else None,
                    viewpoint_name=normalize_waypoint_name(None, capture_count))
                if status is None:
                    break
                capture_mode = resolve_capture_mode(get_gripper_state(), capture_mode_override)
                wp = {
                    "pose_index": capture_count,
                    "name": normalize_waypoint_name(None, capture_count),
                    "capture_joints": get_joints(),
                    "capture_tcp": tcp,
                    "station_index": set_index,
                    "set_index": set_index,
                    "capture_mode": capture_mode,
                    "gripper_state": get_gripper_state(),
                }
                if last_place_joints is not None:
                    wp["place_joints"] = last_place_joints
                if last_place_tcp is not None:
                    wp["place_tcp"] = last_place_tcp
                if set_index >= 0 and set_index in station_records:
                    rec = station_records[set_index]
                    rec['capture_modes_seen'].append(capture_mode)
                    rec['waypoints'].append(wp)
                else:
                    unassigned_waypoints.append(wp)
                    print '  [WARN] Capture stored without station metadata.'
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

            # Stop / abort (replay 가 돌고있지 않을 때 즉시 호출)
            elif cl == 'stop' or cl == 'x':
                stop_flag.set()
                print '[stop] flag set + rb.stop() (soft, decelerating)...'
                try:
                    rb.stop()
                except Exception as exc:
                    print '[stop] rb.stop() failed: {}'.format(exc)
            elif cl == 'abort' or cl == 'xx':
                stop_flag.set()
                abort_flag.set()
                print '[abort] flag set + rb.abort() (hard, motion interrupted)...'
                try:
                    rb.abort()
                except Exception as exc:
                    print '[abort] rb.abort() failed: {}'.format(exc)

            # Auto replay  (in-memory current station OR a JSON file)
            elif cl == 'auto' or cl.startswith('auto'):
                parts = cmd.split()
                file_arg = None
                speed_arg = 30
                # `auto`              -> 현재 station replay
                # `auto 50`           -> 현재 station, speed 50
                # `auto file.json`    -> 파일 replay
                # `auto file.json 50` -> 파일 replay, speed 50
                for tok in parts[1:]:
                    if tok.endswith('.json'):
                        file_arg = tok
                    else:
                        try:
                            speed_arg = int(tok)
                        except ValueError:
                            print '[auto] ignored token: {}'.format(tok)

                # 플래그 초기화 + stdin listener 시작 (auto 동안만 stdin 점유)
                stop_flag.clear()
                abort_flag.clear()
                listener = threading.Thread(target=_stdin_stop_listener, args=(rb,))
                listener.daemon = True
                listener.start()
                try:
                    if file_arg:
                        run_auto_capture(rb, conn, file_arg, speed_arg)
                    else:
                        if set_index < 0 or set_index not in station_records:
                            print '[auto] No current station. Either `set` first + `c` a few times, or pass a JSON file.'
                        else:
                            inline_replay_current_station(
                                rb, conn,
                                station_id=set_index,
                                station_rec=station_records[set_index],
                                speed=speed_arg,
                            )
                finally:
                    # listener 종료 유도: stop_flag 가 set 안돼있으면 set 해서
                    # listener 가 select 다음 회차에 깨어나 빠지게 함.
                    stop_flag.set()
                    listener.join(timeout=0.5)
                    stop_flag.clear()
                    abort_flag.clear()
                    _drain_stdin()

            else:
                print 'Unknown: {}'.format(cmd)

        # Save waypoints
        saved_waypoint_files = save_station_waypoint_files(station_records)
        if saved_waypoint_files:
            print '\nWaypoint files saved:'
            for path in saved_waypoint_files:
                print '  {}'.format(path)
        if unassigned_waypoints:
            with open('object_capture_waypoints_unassigned.json', 'w') as f:
                json.dump({"waypoints": unassigned_waypoints}, f, indent=2)
            print '  object_capture_waypoints_unassigned.json'

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
