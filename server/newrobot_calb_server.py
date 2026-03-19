# newrobot_calb_server.py
from __future__ import print_function

"""
Zeus Robot Calibration Server (new cycle protocol)

Client protocol (per cycle):
  1) server -> client:
       {"command":"capture_cycle", "cycle_index":k}
  2) client -> server:
       {
         "action":"capture_cycle",
         "place_pose_6dof":[x,y,z,rz,ry,rx],
         "capture_pose_6dof":[x,y,z,rz,ry,rx]
       }
  3) server: move place -> move capture
  4) server -> client:
       {
         "event":"cycle_done",
         "cycle_index":k,
         "tcp_pose_6dof":[...]
       }

Notes
-----
- Keeps compatibility with legacy action="capture" + d1..d6 format.
- Uses same i611 move style as existing script (Joint(...), rb.move).
"""

from i611_extend import *
from rbsys import *
from i611_common import *
from i611_io import *
from i611shm import *
import json
import socket
import time


HOST = '0.0.0.0'
PORT = 12348
position_list = []
rb = None
rbs = None

# Motion timing (seconds)
PLACE_SETTLE_SEC = 0.20
CAPTURE_SETTLE_SEC = 0.20
SOCKET_RECV_TIMEOUT_SEC = 30.0


def safe_rb_exit():
    try:
        if rb is not None:
            rb.exit(0)
    except Exception:
        pass


def safe_rb_close():
    try:
        if rb is not None:
            rb.close()
    except Exception:
        pass


def safe_rbs_close():
    try:
        if rbs is not None:
            rbs.close()
    except Exception:
        pass


def safe_rbs_reset():
    try:
        if rbs is not None:
            rbs.cmd_reset()
    except Exception:
        pass


def send_json_to_client(conn, obj):
    if conn is None:
        print('[ERROR] Connection is None')
        return
    try:
        msg = json.dumps(obj)
        conn.sendall(msg.encode('utf-8'))
        print('[TX] {}'.format(msg))
    except socket.error as e:
        print('[ERROR] send failed: {}'.format(e))


def recv_json_from_client(conn, timeout=None):
    old_timeout = None
    try:
        old_timeout = conn.gettimeout()
    except Exception:
        pass

    try:
        if timeout is not None:
            conn.settimeout(float(timeout))

        data = conn.recv(4096)
        if not data:
            return None

        txt = data.decode('utf-8').strip()
        if txt == '':
            return None

        try:
            obj = json.loads(txt)
            print('[RX] {}'.format(obj))
            return obj
        except ValueError:
            print('[WARN] JSON decode failed. raw={}'.format(txt))
            return {'raw': txt}

    except socket.timeout:
        return None
    except socket.error as e:
        print('[ERROR] recv failed: {}'.format(e))
        return None
    finally:
        if timeout is not None:
            try:
                conn.settimeout(old_timeout)
            except Exception:
                pass


def parse_pose6(obj):
    if obj is None:
        return None

    if isinstance(obj, list) and len(obj) == 6:
        try:
            return [float(x) for x in obj]
        except Exception:
            return None

    if isinstance(obj, dict):
        keys = ['x', 'y', 'z', 'rz', 'ry', 'rx']
        if all((k in obj) for k in keys):
            try:
                return [
                    float(obj['x']), float(obj['y']), float(obj['z']),
                    float(obj['rz']), float(obj['ry']), float(obj['rx'])
                ]
            except Exception:
                return None

        for k in ['pose_6dof', 'capture_pose_6dof', 'place_pose_6dof', 'tcp_pose_6dof', 'pose']:
            if k in obj:
                p = parse_pose6(obj[k])
                if p is not None:
                    return p

    return None


def pose6_to_joint(p):
    return Joint(p[0], p[1], p[2], p[3], p[4], p[5])


def move_pose6(pose6, label='pose'):
    tgt = pose6_to_joint(pose6)
    print('[MOVE] {} -> {}'.format(label, pose6))
    rb.move(tgt)
    print('[MOVE] {} done'.format(label))


def get_curr_tcp_pose():
    pose = rb.getpos()
    vals = pose.pos2list()
    x = vals[0]
    y = vals[1]
    z = vals[2]
    rz = vals[3]
    ry = vals[4]
    rx = vals[5]
    return [x, y, z, rz, ry, rx]


def record_curr_position():
    global position_list
    tcp = get_curr_tcp_pose()
    position_list.append(tcp)
    print('[INFO] Current TCP: {}'.format(tcp))
    return tcp


def send_capture_cycle_request(conn, cycle_index):
    send_json_to_client(conn, {
        'command': 'capture_cycle',
        'cycle_index': int(cycle_index),
    })


def send_quit_command(conn):
    send_json_to_client(conn, {'command': 'quit'})


def maybe_reply_tcp_pose_query(conn):
    msg = recv_json_from_client(conn, timeout=0.20)
    if not isinstance(msg, dict):
        return

    action = str(msg.get('action', '')).strip().lower()
    if action == 'get_tcp_pose':
        tcp = get_curr_tcp_pose()
        send_json_to_client(conn, {'tcp_pose_6dof': tcp})


def handle_cycle_message(conn, msg, fallback_cycle_index):
    if not isinstance(msg, dict):
        return False

    action = str(msg.get('action', '')).strip().lower()

    if action in ['stop', 'quit', 'end']:
        print('[INFO] Stop requested by client')
        send_quit_command(conn)
        return None

    if action == 'get_tcp_pose':
        tcp = get_curr_tcp_pose()
        send_json_to_client(conn, {'tcp_pose_6dof': tcp})
        return False

    cycle_index = int(msg.get('cycle_index', fallback_cycle_index))

    if action == 'capture_cycle':
        place_pose = parse_pose6(msg.get('place_pose_6dof'))
        capture_pose = parse_pose6(msg.get('capture_pose_6dof'))

        if capture_pose is None:
            send_json_to_client(conn, {
                'event': 'error',
                'reason': 'capture_pose_6dof is missing/invalid',
                'cycle_index': cycle_index,
            })
            return False

        if place_pose is None:
            place_pose = capture_pose

        move_pose6(place_pose, label='place_pose')
        if PLACE_SETTLE_SEC > 0:
            time.sleep(PLACE_SETTLE_SEC)

        move_pose6(capture_pose, label='capture_pose')
        if CAPTURE_SETTLE_SEC > 0:
            time.sleep(CAPTURE_SETTLE_SEC)

        tcp = record_curr_position()

        send_json_to_client(conn, {
            'event': 'cycle_done',
            'cycle_index': cycle_index,
            'tcp_pose_6dof': tcp,
            'place_pose_6dof': place_pose,
            'capture_pose_6dof': capture_pose,
        })
        maybe_reply_tcp_pose_query(conn)
        return True

    if action == 'capture':
        # Legacy one-pose mode
        try:
            p = [
                float(msg['d1']), float(msg['d2']), float(msg['d3']),
                float(msg['d4']), float(msg['d5']), float(msg['d6'])
            ]
        except Exception:
            send_json_to_client(conn, {
                'event': 'error',
                'reason': 'legacy capture needs d1..d6',
                'cycle_index': cycle_index,
            })
            return False

        move_pose6(p, label='legacy_capture_pose')
        if CAPTURE_SETTLE_SEC > 0:
            time.sleep(CAPTURE_SETTLE_SEC)

        tcp = record_curr_position()
        send_json_to_client(conn, {
            'event': 'cycle_done',
            'cycle_index': cycle_index,
            'tcp_pose_6dof': tcp,
            'capture_pose_6dof': p,
        })
        maybe_reply_tcp_pose_query(conn)
        return True

    send_json_to_client(conn, {
        'event': 'error',
        'reason': 'unknown action',
        'action': action,
        'cycle_index': cycle_index,
    })
    return False


def main(conn):
    global position_list

    try:
        # Robot motion setup
        m = MotionParam(
            jnt_speed=70,
            lin_speed=50,
            pose_speed=50,
            overlap=0,
            acctime=1.0,
            dacctime=1.0,
        )
        rb.motionparam(m)
        rb.override(50)

        rb.settool(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rb.settool(2, 0.0, 35.0, 330.0, 0.0, 0.0, 0.0)
        rb.settool(3, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0)
        rb.changetool(3)
        rb.use_mt(True)

        conn.settimeout(SOCKET_RECV_TIMEOUT_SEC)

        cycle_count = 0
        while True:
            send_capture_cycle_request(conn, cycle_count)

            msg = recv_json_from_client(conn)
            if msg is None:
                print('[INFO] No client message. stopping server loop')
                send_quit_command(conn)
                break

            handled = handle_cycle_message(conn, msg, cycle_count)

            if handled is None:
                # stop requested
                break

            if handled:
                cycle_count += 1
                print('[INFO] Completed cycle #{}'.format(cycle_count))

    except Robot_emo as e:
        print(e)
        safe_rb_exit()
        safe_rbs_reset()

    except Robot_error as e:
        print(e)
        safe_rb_exit()
        safe_rbs_reset()

    except Robot_fatalerror as e:
        print(e)
        safe_rb_exit()
        safe_rbs_reset()

    except Exception as e:
        print(e)
        safe_rb_exit()

    except KeyboardInterrupt:
        safe_rb_exit()
        print('Key Interrupt')

    finally:
        print('[INFO] position_list size={}'.format(len(position_list)))
        safe_rb_close()
        safe_rbs_close()
        safe_rb_exit()


derom i611_extend import *
from rbsys import *
from i611_common import *
from i611_io import *
from i611shm import *
import json
import socket
import time


HOST = '0.0.0.0'
PORT = 12348
position_list = []
rb = None
rbs = None

# Motion timing (seconds)
PLACE_SETTLE_SEC = 0.20
CAPTURE_SETTLE_SEC = 0.20
SOCKET_RECV_TIMEOUT_SEC = 30.0


def safe_rb_exit():
    try:
        if rb is not None:
            rb.exit(0)
    except Exception:
        pass


def safe_rb_close():
    try:
        if rb is not None:
            rb.close()
    except Exception:
        pass


def safe_rbs_close():
    try:
        if rbs is not None:
            rbs.close()
    except Exception:
        pass


def safe_rbs_reset():
    try:
        if rbs is not None:
            rbs.cmd_reset()
    except Exception:
        pass


def send_json_to_client(conn, obj):
    if conn is None:
        print('[ERROR] Connection is None')
        return
    try:
        msg = json.dumps(obj)
        conn.sendall(msg.encode('utf-8'))
        print('[TX] {}'.format(msg))
    except socket.error as e:
        print('[ERROR] send failed: {}'.format(e))


def recv_json_from_client(conn, timeout=None):
    old_timeout = None
    try:
        old_timeout = conn.gettimeout()
    except Exception:
        pass

    try:
        if timeout is not None:
            conn.settimeout(float(timeout))

        data = conn.recv(4096)
        if not data:
            return None

        txt = data.decode('utf-8').strip()
        if txt == '':
            return None

        try:
            obj = json.loads(txt)
            print('[RX] {}'.format(obj))
            return obj
        except ValueError:
            print('[WARN] JSON decode failed. raw={}'.format(txt))
            return {'raw': txt}

    except socket.timeout:
        return None
    except socket.error as e:
        print('[ERROR] recv failed: {}'.format(e))
        return None
    finally:
        if timeout is not None:
            try:
                conn.settimeout(old_timeout)
            except Exception:
                pass


def parse_pose6(obj):
    if obj is None:
        return None

    if isinstance(obj, list) and len(obj) == 6:
        try:
            return [float(x) for x in obj]
        except Exception:
            return None

    if isinstance(obj, dict):
        keys = ['x', 'y', 'z', 'rz', 'ry', 'rx']
        if all((k in obj) for k in keys):
            try:
                return [
                    float(obj['x']), float(obj['y']), float(obj['z']),
                    float(obj['rz']), float(obj['ry']), float(obj['rx'])
                ]
            except Exception:
                return None

        for k in ['pose_6dof', 'capture_pose_6dof', 'place_pose_6dof', 'tcp_pose_6dof', 'pose']:
            if k in obj:
                p = parse_pose6(obj[k])
                if p is not None:
                    return p

    return None


def pose6_to_joint(p):
    return Joint(p[0], p[1], p[2], p[3], p[4], p[5])


def move_pose6(pose6, label='pose'):
    tgt = pose6_to_joint(pose6)
    print('[MOVE] {} -> {}'.format(label, pose6))
    rb.move(tgt)
    print('[MOVE] {} done'.format(label))


def get_curr_tcp_pose():
    pose = rb.getpos()
    vals = pose.pos2list()
    x = vals[0]
    y = vals[1]
    z = vals[2]
    rz = vals[3]
    ry = vals[4]
    rx = vals[5]
    return [x, y, z, rz, ry, rx]


def record_curr_position():
    global position_list
    tcp = get_curr_tcp_pose()
    position_list.append(tcp)
    print('[INFO] Current TCP: {}'.format(tcp))
    return tcp


def send_capture_cycle_request(conn, cycle_index):
    send_json_to_client(conn, {
        'command': 'capture_cycle',
        'cycle_index': int(cycle_index),
    })


def send_quit_command(conn):
    send_json_to_client(conn, {'command': 'quit'})


def maybe_reply_tcp_pose_query(conn):
    msg = recv_json_from_client(conn, timeout=0.20)
    if not isinstance(msg, dict):
        return

    action = str(msg.get('action', '')).strip().lower()
    if action == 'get_tcp_pose':
        tcp = get_curr_tcp_pose()
        send_json_to_client(conn, {'tcp_pose_6dof': tcp})


def handle_cycle_message(conn, msg, fallback_cycle_index):
    if not isinstance(msg, dict):
        return False

    action = str(msg.get('action', '')).strip().lower()

    if action in ['stop', 'quit', 'end']:
        print('[INFO] Stop requested by client')
        send_quit_command(conn)
        return None

    if action == 'get_tcp_pose':
        tcp = get_curr_tcp_pose()
        send_json_to_client(conn, {'tcp_pose_6dof': tcp})
        return False

    cycle_index = int(msg.get('cycle_index', fallback_cycle_index))

    if action == 'capture_cycle':
        place_pose = parse_pose6(msg.get('place_pose_6dof'))
        capture_pose = parse_pose6(msg.get('capture_pose_6dof'))

        if capture_pose is None:
            send_json_to_client(conn, {
                'event': 'error',
                'reason': 'capture_pose_6dof is missing/invalid',
                'cycle_index': cycle_index,
            })
            return False

        if place_pose is None:
            place_pose = capture_pose

        move_pose6(place_pose, label='place_pose')
        if PLACE_SETTLE_SEC > 0:
            time.sleep(PLACE_SETTLE_SEC)

        move_pose6(capture_pose, label='capture_pose')
        if CAPTURE_SETTLE_SEC > 0:
            time.sleep(CAPTURE_SETTLE_SEC)

        tcp = record_curr_position()

        send_json_to_client(conn, {
            'event': 'cycle_done',
            'cycle_index': cycle_index,
            'tcp_pose_6dof': tcp,
            'place_pose_6dof': place_pose,
            'capture_pose_6dof': capture_pose,
        })
        maybe_reply_tcp_pose_query(conn)
        return True

    if action == 'capture':
        # Legacy one-pose mode
        try:
            p = [
                float(msg['d1']), float(msg['d2']), float(msg['d3']),
                float(msg['d4']), float(msg['d5']), float(msg['d6'])
            ]
        except Exception:
            send_json_to_client(conn, {
                'event': 'error',
                'reason': 'legacy capture needs d1..d6',
                'cycle_index': cycle_index,
            })
            return False

        move_pose6(p, label='legacy_capture_pose')
        if CAPTURE_SETTLE_SEC > 0:
            time.sleep(CAPTURE_SETTLE_SEC)

        tcp = record_curr_position()
        send_json_to_client(conn, {
            'event': 'cycle_done',
            'cycle_index': cycle_index,
            'tcp_pose_6dof': tcp,
            'capture_pose_6dof': p,
        })
        maybe_reply_tcp_pose_query(conn)
        return True

    send_json_to_client(conn, {
        'event': 'error',
        'reason': 'unknown action',
        'action': action,
        'cycle_index': cycle_index,
    })
    return False


def main(conn):
    global position_list

    try:
        # Robot motion setup
        m = MotionParam(
            jnt_speed=70,
            lin_speed=50,
            pose_speed=50,
            overlap=0,
            acctime=1.0,
            dacctime=1.0,
        )
        rb.motionparam(m)
        rb.override(50)

        rb.settool(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rb.settool(2, 0.0, 35.0, 330.0, 0.0, 0.0, 0.0)
        rb.settool(3, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0)
        rb.changetool(3)
        rb.use_mt(True)

        conn.settimeout(SOCKET_RECV_TIMEOUT_SEC)

        cycle_count = 0
        while True:
            send_capture_cycle_request(conn, cycle_count)

            msg = recv_json_from_client(conn)
            if msg is None:
                print('[INFO] No client message. stopping server loop')
                send_quit_command(conn)
                break

            handled = handle_cycle_message(conn, msg, cycle_count)

            if handled is None:
                # stop requested
                break

            if handled:
                cycle_count += 1
                print('[INFO] Completed cycle #{}'.format(cycle_count))

    except Robot_emo as e:
        print(e)
        safe_rb_exit()
        safe_rbs_reset()

    except Robot_error as e:
        print(e)
        safe_rb_exit()
        safe_rbs_reset()

    except Robot_fatalerror as e:
        print(e)
        safe_rb_exit()
        safe_rbs_reset()

    except Exception as e:
        print(e)
        safe_rb_exit()

    except KeyboardInterrupt:
        safe_rb_exit()
        print('Key Interrupt')

    finally:
        print('[INFO] position_list size={}'.format(len(position_list)))
        safe_rb_close()
        safe_rbs_close()
        safe_rb_exit()


def start_server():
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)

        print('Server started. Waiting for client connection...')
        conn, addr = s.accept()
        print('Client connected: {}'.format(addr))

        main(conn)

    except socket.error as e:
        print('Socket error: {}'.format(e))
    finally:
        if s is not None:
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
        safe_rb_exit()

    except Robot_emo:
        safe_rb_exit()
        safe_rbs_reset()

    except Robot_error:
        safe_rb_exit()
        safe_rbs_reset()

    except Robot_fatalerror:
        safe_rb_exit()
        safe_rbs_reset()

    finally:
        safe_rb_close()
        safe_rbs_close()
        safe_rb_exit()
f start_server():
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)

        print('Server started. Waiting for client connection...')
        conn, addr = s.accept()
        print('Client connected: {}'.format(addr))

        main(conn)

    except socket.error as e:
        print('Socket error: {}'.format(e))
    finally:
        if s is not None:
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
        safe_rb_exit()

    except Robot_emo:
        safe_rb_exit()
        safe_rbs_reset()

    except Robot_error:
        safe_rb_exit()
        safe_rbs_reset()

    except Robot_fatalerror:
        safe_rb_exit()
        safe_rbs_reset()

    finally:
        safe_rb_close()
        safe_rbs_close()
        safe_rb_exit()
