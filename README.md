# ViLock - Face Detection Screen Lock for macOS

Automatically lock your MacBook screen when no face is detected via camera.

## Features

- **Face Detection** - MediaPipe with 95%+ accuracy
- **Face Verification** - Auto-unlock when registered face detected
- **Telegram Notifications** - Get notified on lock/unlock events
- **Menu Bar App** - Runs in background with menu bar controls

## Requirements

- macOS 10.14+
- Python 3.9+
- Camera access permission

## Installation

```bash
git clone https://github.com/hungnm58/ViLock.git
cd ViLock
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
source venv/bin/activate
python -m src.app
```

## Menu Bar Controls

| Feature | Description |
|---------|-------------|
| Start/Stop | Toggle face detection monitoring |
| Register Face | Register face for auto-unlock |
| Set Password | Set unlock password |
| Telegram Config | Configure Telegram notifications |
| Lock Now | Lock screen immediately |

## Configuration

Data stored in `./data/`:

| File | Content |
|------|---------|
| `settings.plist` | App configuration |
| `faces/registered_face.jpg` | Registered face image |
| `faces/face_encoding.npy` | Face encoding vector |

## Project Structure

```
ViLock/
├── src/
│   ├── app.py              # Entry point
│   ├── menu_bar.py         # Menu bar app (rumps)
│   ├── detector.py         # MediaPipe face detection
│   ├── face_verifier.py    # Face verification
│   ├── notifier.py         # Telegram notifications
│   └── settings.py         # Settings persistence
├── data/                   # User data (gitignored)
├── assets/                 # ML models
└── requirements.txt
```

## License

MIT
