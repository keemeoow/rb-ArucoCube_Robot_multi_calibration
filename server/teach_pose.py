#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Teach pose tool: manually jog the robot and save the current TCP pose.

Commands:
  p <axis>,<value>  : TCP move (e.g., "p z,50" moves z+50mm)
  j <axis>,<value>  : Joint move (e.g., "j d1,10" moves J1+10deg)
  show              : Print current TCP pose
  save              : Print current TCP pose as seed_place format (copy-paste)
  speed <0-100>     : Change override speed
  q                 : Quit

Usage:
  python teach_pose.py
"""

from i611_MCS import *
from teachdata import *
from i611_extend import *
from rbsys import *
from i611_common import *
from i611_io import *
from i611shm import *


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
    print ''
    print '=== Current Joints ==='
    print '  d1={:.3f}  d2={:.3f}  d3={:.3f}'.format(jnt[0], jnt[1], jnt[2])
    print '  d4={:.3f}  d5={:.3f}  d6={:.3f}'.format(jnt[3], jnt[4], jnt[5])
    print ''
    return tcp


def show_seed_format():
    tcp = get_tcp()
    print ''
    print '>>> Copy this as --seed_place:'
    print '    {:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}'.format(
        tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])
    print ''
    print '>>> Full command:'
    print '    python new2_make_zeus_calib_waypoints.py \\'
    print '      --seed_place {:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f} \\'.format(
        tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])
    print '      --capture_z_offset 200 \\'
    print '      --out_file new2_waypoints.json'
    print ''


def move_tcp(axis, value):
    pose = rb.getpos()
    vals = pose.pos2list()
    x, y, z, rz, ry, rx = vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
    current = Position(x, y, z, rz, ry, rx)

    if axis == 'x':
        target = current.offset(dx=value)
    elif axis == 'y':
        target = current.offset(dy=value)
    elif axis == 'z':
        target = current.offset(dz=value)
    elif axis == 'rz':
        target = current.offset(drz=value)
    elif axis == 'ry':
        target = current.offset(dry=value)
    elif axis == 'rx':
        target = current.offset(drx=value)
    else:
        print 'Invalid axis: {}. Use x,y,z,rz,ry,rx'.format(axis)
        return

    print 'TCP move: {} += {}'.format(axis, value)
    rb.line(target)
    show_pose()


def move_joint(axis, value):
    jnt = rb.getjnt()
    vals = jnt.jnt2list()
    current = Joint(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])

    if axis == 'd1':
        target = current.offset(dj1=value)
    elif axis == 'd2':
        target = current.offset(dj2=value)
    elif axis == 'd3':
        target = current.offset(dj3=value)
    elif axis == 'd4':
        target = current.offset(dj4=value)
    elif axis == 'd5':
        target = current.offset(dj5=value)
    elif axis == 'd6':
        target = current.offset(dj6=value)
    else:
        print 'Invalid axis: {}. Use d1,d2,d3,d4,d5,d6'.format(axis)
        return

    print 'Joint move: {} += {}'.format(axis, value)
    rb.move(target)
    show_pose()


def main():
    try:
        rbs = RobSys()
        rbs.open()
        rb_local = i611Robot()
        _BASE = Base()
        rb_local.open()

        global rb
        rb = rb_local

        m = MotionParam(jnt_speed=30, lin_speed=50, pose_speed=50,
                        overlap=0, acctime=0.8, dacctime=0.8)
        rb.motionparam(m)
        rb.override(30)

        rb.settool(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rb.settool(2, 0.0, 35.0, 330.0, 0.0, 0.0, 0.0)
        rb.settool(3, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0)
        rb.changetool(3)

        print ''
        print '=============================='
        print '  Teach Pose Tool'
        print '=============================='
        print 'Commands:'
        print '  p <axis>,<value>  : TCP move (e.g., "p z,50")'
        print '  j <axis>,<value>  : Joint move (e.g., "j d1,10")'
        print '  show              : Show current pose'
        print '  save              : Show pose as seed_place format'
        print '  speed <0-100>     : Set speed override'
        print '  q                 : Quit'
        print ''

        show_pose()

        while True:
            try:
                cmd = raw_input('> ').strip()
            except EOFError:
                break

            if not cmd:
                continue

            if cmd.lower() == 'q':
                break

            elif cmd.lower() == 'show':
                show_pose()

            elif cmd.lower() == 'save':
                show_seed_format()

            elif cmd.lower().startswith('speed'):
                try:
                    spd = int(cmd.split()[1])
                    rb.override(spd)
                    print 'Speed set to {}'.format(spd)
                except Exception:
                    print 'Usage: speed <0-100>'

            elif cmd.lower().startswith('p '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis = parts[0].strip()
                    value = float(parts[1].strip())
                    move_tcp(axis, value)
                except Exception as e:
                    print 'Error: {}. Usage: p <axis>,<value>'.format(e)

            elif cmd.lower().startswith('j '):
                try:
                    parts = cmd[2:].strip().split(',')
                    axis = parts[0].strip()
                    value = float(parts[1].strip())
                    move_joint(axis, value)
                except Exception as e:
                    print 'Error: {}. Usage: j <axis>,<value>'.format(e)

            else:
                print 'Unknown command: {}'.format(cmd)

    except KeyboardInterrupt:
        print '\nInterrupted'
    except Exception as e:
        print 'Error: {}'.format(e)
    finally:
        try:
            rb.exit(0)
            rb.close()
            rbs.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
