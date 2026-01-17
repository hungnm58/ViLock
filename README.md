# AutoLock - macOS Auto Screen Lock

Automatically locks your Mac screen when no face is detected via camera.

## Features

- **MediaPipe Face Detection** - 95%+ accuracy, works with angles and low light
- **Menu Bar App** - Runs in background with quick controls
- **Preferences Window** - Configure timeout, sensitivity, camera
- **Settings Persistence** - Settings saved across restarts
- **Auto-Start** - Optional launch at login
- **Packagable** - Can be bundled as .app with py2app

## Requirements

- macOS 10.14+
- Python 3.9+
- Camera access permission

## Installation

### 1. Clone and setup environment

```bash
cd autolock
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Grant camera permission

When first run, macOS will prompt for camera access. Go to:
**System Preferences → Privacy & Security → Camera** → Allow Terminal/Python

## Usage

### Run menu bar app

```bash
source venv/bin/activate
python -m src.app
```

Or with the old CLI version:
```bash
python auto_lock.py -d  # debug mode with camera preview
```

### Menu Bar Controls

- **Start/Stop Monitoring** - Toggle face detection
- **Pause/Resume** - Temporarily pause without stopping
- **Timeout** - Set lock delay (5-60 seconds)
- **Preferences** - Open settings window
- **Lock Now** - Immediate lock

### Settings

Settings stored at: `~/Library/Preferences/com.autolock.plist`

| Setting | Default | Description |
|---------|---------|-------------|
| timeout | 10s | Seconds without face before lock |
| confidence_threshold | 0.5 | Detection sensitivity (0.3-0.9) |
| camera_index | 0 | Camera device to use |
| start_at_login | false | Auto-start when macOS boots |

## Build .app Bundle

```bash
pip install py2app
python setup.py py2app
# Output: dist/AutoLock.app
```

## Project Structure

```
autolock/
├── src/
│   ├── app.py          # Entry point
│   ├── detector.py     # MediaPipe face detection
│   ├── menu_bar.py     # Rumps menu bar app
│   ├── preferences.py  # Tkinter settings window
│   ├── settings.py     # Plistlib persistence
│   └── autostart.py    # LaunchAgent management
├── assets/
│   └── blaze_face_short_range.tflite
├── auto_lock.py        # Legacy CLI version
├── setup.py            # py2app config
└── requirements.txt
```

## Troubleshooting

### Camera not working
```bash
python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```
Check camera permissions in System Preferences.

### Detection inaccurate
- Increase lighting
- Adjust `confidence_threshold` in Preferences (lower = more sensitive)
- Ensure face is visible to camera

### App doesn't start at login
Run the app once and enable "Start at login" in Preferences. This creates a LaunchAgent at `~/Library/LaunchAgents/com.autolock.plist`.

## License

MIT
