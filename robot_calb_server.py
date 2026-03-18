# zeus_calib_server.py
"""
Zeus Robot Calibration Server (i611 SDK)
Step2_capture_cube_poses.py 클라이언트와 TCP 소켓으로 통신하는 서버.

기존 server.py 대비 변경점:
  - "capture" 대신 JSON {"command":"capture", "tcp_pose_6dof":[...]} 전송
  - 클라이언트의 "get_tcp_pose" 쿼리 응답 지원
  - 로봇 이동 후 현재 TCP pose를 함께 기록

프로토콜 흐름 (각 waypoint마다):
  1. 서버 → 클라이언트: {"command":"capture", "tcp_pose_6dof":[x,y,z,rz,ry,rx]}
  2. 클라이언트 → 서버: {"action":"capture", "d1":..., ..., "d6":...}
  3. 서버: d1~d6 으로 로봇 이동 (rb.move)
  4. (선택적) 클라이언트 → 서버: {"action":"get_tcp_pose"}
  5. (선택적) 서버 → 클라이언트: {"tcp_pose_6dof":[x,y,z,rz,ry,rx]}
"""

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
position_list = []


# ──────────────────────────────────────────────────────────────
# 소켓 통신
# ──────────────────────────────────────────────────────────────

def send_json_to_client(conn, obj):
    """JSON 객체를 클라이언트에 전송."""
    if conn is None:
        print("[ERROR] Connection is None")
        return
    try:
        msg = json.dumps(obj)
        conn.sendall(msg.encode('utf-8'))
        print "Sent to client: {}".format(msg)
    except socket.error as e:
        print "Error sending to client: {}".format(e)


def receive_json_from_client(conn):
    """클라이언트로부터 JSON 메시지 수신."""
    try:
        data = conn.recv(4096).decode('utf-8')
        if data:
            data = data.strip()
            try:
                received = json.loads(data)
                print "Received from client: {}".format(received)
                return received
            except ValueError as e:
                print "JSON decode error: {}".format(e)
                return data
    except socket.error as e:
        print "Error receiving from client: {}".format(e)
    return None


# ──────────────────────────────────────────────────────────────
# 로봇 포즈
# ──────────────────────────────────────────────────────────────

def get_curr_tcp_pose():
    """현재 로봇 TCP 포즈를 [x, y, z, rz, ry, rx] 리스트로 반환."""
    pose = rb.getpos()
    vals = pose.pos2list()
    x  = vals[0]
    y  = vals[1]
    z  = vals[2]
    rz = vals[3]
    ry = vals[4]
    rx = vals[5]
    return [x, y, z, rz, ry, rx]


def get_curr_position():
    """현재 위치를 position_list에 기록하고 반환."""
    global position_list
    tcp = get_curr_tcp_pose()
    position_list.append(tcp)
    print "Current TCP: {}".format(tcp)
    return position_list


# ──────────────────────────────────────────────────────────────
# 캘리브레이션 명령 사이클
# ──────────────────────────────────────────────────────────────

def send_capture_command(conn):
    """
    "capture" 명령 + 현재 TCP pose를 클라이언트에 전송하고,
    클라이언트가 보낸 관절 명령(d1~d6)을 수신하여 반환.
    """
    # 현재 TCP pose 포함하여 capture 명령 전송
    tcp_pose = get_curr_tcp_pose()
    cmd = {
        "command": "capture",
        "tcp_pose_6dof": tcp_pose
    }
    send_json_to_client(conn, cmd)

    # 클라이언트 응답 수신
    received = receive_json_from_client(conn)
    if received and isinstance(received, dict):
        action = received.get('action', '')

        if action == 'capture':
            goal_joint = [
                received['d1'], received['d2'], received['d3'],
                received['d4'], received['d5'], received['d6']
            ]
            print 'goal_joint: {}'.format(goal_joint)
            return goal_joint

    return None


def handle_tcp_pose_query(conn):
    """
    클라이언트의 get_tcp_pose 쿼리를 처리.
    settle_time 후 클라이언트가 TCP pose를 요청할 수 있음.
    """
    try:
        conn.settimeout(3.0)
        data = conn.recv(4096).decode('utf-8').strip()
        if data:
            try:
                msg = json.loads(data)
                if isinstance(msg, dict) and msg.get('action') == 'get_tcp_pose':
                    tcp_pose = get_curr_tcp_pose()
                    print "TCP pose query response: {}".format(tcp_pose)
                    send_json_to_client(conn, {"tcp_pose_6dof": tcp_pose})
            except ValueError:
                pass
    except socket.timeout:
        pass  # 쿼리 없음 — 정상
    except socket.error:
        pass


def send_quit_command(conn):
    """종료 명령 전송."""
    send_json_to_client(conn, {"command": "quit"})
    return 0


# ──────────────────────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────────────────────

def main(conn):
    global position_list
    try:
        m = MotionParam(jnt_speed=70, lin_speed=50, pose_speed=50,
                        overlap=0, acctime=1.0, dacctime=1.0)
        rb.motionparam(m)
        rb.override(50)

        rb.settool(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rb.settool(2, 0.0, 35.0, 330.0, 0.0, 0.0, 0.0)
        rb.settool(3, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0)
        rb.changetool(3)
        rb.use_mt(True)

        capture_count = 0

        while True:
            # 1. capture 명령 전송 + 관절 명령 수신
            goal = send_capture_command(conn)
            print 'goal joint is: {}'.format(goal)

            if goal is None:
                print 'EOF — no valid joint received'
                send_quit_command(conn)
                print 'Position list: {}'.format(position_list)
                break

            # 2. 관절 이동
            goal_jnt = Joint(goal[0], goal[1], goal[2],
                             goal[3], goal[4], goal[5])
            print 'Moving to goal joint...'
            rb.move(goal_jnt)
            print 'Move complete'

            # 3. 이동 후 현재 위치 기록
            get_curr_position()
            capture_count += 1
            print 'Capture #{} done'.format(capture_count)

            # 4. 클라이언트의 get_tcp_pose 쿼리 처리 (선택적)
            handle_tcp_pose_query(conn)

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
# 서버 시작
# ──────────────────────────────────────────────────────────────

def start_server():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print "Server started. Waiting for client connection..."

        conn, addr = s.accept()
        print("Client connected: ", addr)
        print "Connected to client {}".format(addr)

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
