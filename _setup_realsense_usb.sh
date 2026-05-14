#!/usr/bin/env bash
# RealSense 멀티카메라 USB 안정화 설정 스크립트.
#   1) USB autosuspend 비활성화 (runtime + GRUB 영구)
#   2) uvcvideo 드라이버 옵션 (nodrop=1, timeout=10000)
#   3) librealsense udev rules 리로드
# 실행: sudo bash scripts/setup_realsense_usb.sh
# 적용: 1·2번은 재부팅 후 영구 적용. 1번 runtime은 즉시 적용.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] root 권한 필요. 'sudo bash $0' 로 실행하세요." >&2
    exit 1
fi

ts=$(date +%Y%m%d_%H%M%S)

echo "==[1/3] USB autosuspend 비활성화=="
# runtime
echo -1 > /sys/module/usbcore/parameters/autosuspend
echo "  [OK] runtime autosuspend = $(cat /sys/module/usbcore/parameters/autosuspend)"

# persistent via GRUB
grub_file=/etc/default/grub
cp "$grub_file" "${grub_file}.bak.${ts}"
echo "  [OK] backup: ${grub_file}.bak.${ts}"

if grep -q "usbcore.autosuspend" "$grub_file"; then
    sed -i 's/usbcore\.autosuspend=[^ "]*/usbcore.autosuspend=-1/g' "$grub_file"
    echo "  [OK] GRUB usbcore.autosuspend 갱신"
else
    sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 usbcore.autosuspend=-1"/' "$grub_file"
    echo "  [OK] GRUB usbcore.autosuspend=-1 추가"
fi
update-grub
echo "  [OK] update-grub 완료"

echo "==[2/3] uvcvideo 드라이버 옵션=="
uvc_conf=/etc/modprobe.d/uvcvideo.conf
cat > "$uvc_conf" <<'EOF'
# RealSense 멀티카메라 안정화
#   nodrop=1   : 손상 프레임 드롭하지 않고 상위로 전달 (멀티 카메라 동기 안정성)
#   timeout=10000 : 프레임 대기 타임아웃 10초로 확장
options uvcvideo nodrop=1 timeout=10000
EOF
echo "  [OK] $uvc_conf 작성"
update-initramfs -u
echo "  [OK] update-initramfs 완료"

echo "==[3/3] librealsense udev rules 리로드=="
if [[ -f /etc/udev/rules.d/99-realsense-libusb.rules ]]; then
    udevadm control --reload-rules
    udevadm trigger
    echo "  [OK] udev rules 리로드 완료"
else
    echo "  [WARN] 99-realsense-libusb.rules 없음. ROS humble librealsense2 설치 확인 필요"
fi

echo ""
echo "=== 완료 ==="
echo "1·2번은 [재부팅] 후 영구 적용됩니다."
echo "현재 세션에서는 autosuspend runtime 값만 즉시 적용됨."
echo "재부팅 후 확인: cat /proc/cmdline | grep autosuspend"
echo "                cat /sys/module/uvcvideo/parameters/nodrop"
