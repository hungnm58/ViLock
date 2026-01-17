# ViLock - Face Detection Screen Lock for macOS

Ứng dụng tự động khóa màn hình MacBook khi không phát hiện khuôn mặt qua camera.

## Tính năng

- **Face Detection** - Sử dụng MediaPipe với độ chính xác 95%+
- **Face Verification** - Xác thực khuôn mặt để mở khóa tự động
- **Bluetooth Unlock** - Mở khóa khi iPhone/thiết bị Bluetooth ở gần
- **Telegram Notifications** - Nhận thông báo khi lock/unlock
- **Menu Bar App** - Chạy ngầm với icon trên thanh menu

## Yêu cầu

- macOS 10.14+
- Python 3.9+
- Camera access permission

## Cài đặt

```bash
git clone https://github.com/user/ViLock.git
cd ViLock
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Sử dụng

```bash
source venv/bin/activate
python -m src.app
```

## Menu Bar Controls

| Chức năng | Mô tả |
|-----------|-------|
| Start/Stop | Bật/tắt theo dõi khuôn mặt |
| Register Face | Đăng ký khuôn mặt để unlock |
| Set Password | Đặt mật khẩu unlock |
| Telegram Config | Cấu hình thông báo Telegram |
| Lock Now | Khóa màn hình ngay lập tức |

## Cấu hình

Dữ liệu lưu trong `./data/`:

| File | Nội dung |
|------|----------|
| `settings.plist` | Cấu hình ứng dụng |
| `faces/registered_face.jpg` | Ảnh khuôn mặt đã đăng ký |
| `faces/face_encoding.npy` | Face encoding vector |

## Cấu trúc project

```
ViLock/
├── src/
│   ├── app.py              # Entry point
│   ├── menu_bar.py         # Menu bar app (rumps)
│   ├── detector.py         # MediaPipe face detection
│   ├── face_verifier.py    # Face verification
│   ├── bluetooth_unlocker.py # Bluetooth proximity unlock
│   ├── notifier.py         # Telegram notifications
│   └── settings.py         # Settings persistence
├── data/                   # User data (gitignored)
├── assets/                 # ML models
└── requirements.txt
```

## License

MIT
