# ViLock - Face Detection Screen Lock for macOS

Automatically lock your MacBook screen when no face is detected via camera.

## Features

- **Face Detection** - MediaPipe with 95%+ accuracy, optimized VIDEO mode
- **Face Verification** - Auto-unlock when registered face detected
- **Encrypted Face Data** - AES-128 encryption, key stored in macOS Keychain
- **Telegram Notifications** - Get notified on lock/unlock events
- **Menu Bar App** - Runs in background with menu bar controls

## Requirements

- macOS 10.14+
- Python 3.9+
- Camera access permission

## Installation

### 1. Clone and setup

```bash
git clone https://github.com/hungnm58/ViLock.git
cd ViLock
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Grant Camera permission

On first run, macOS will prompt for Camera access:
- **System Settings → Privacy & Security → Camera** → Enable for Terminal/iTerm

## Usage

### Run the app

```bash
cd /path/to/ViLock
source venv/bin/activate
python -m src.app
```

### Menu Bar Controls

Icon 🔓 appears in menu bar:

| Menu | Function |
|------|----------|
| **Start Monitoring** | Start face detection |
| **Pause** | Pause monitoring |
| **Timeout** | Lock delay time (1-60s) |
| **🔑 Set Password** | Set unlock password |
| **👤 Register Face** | Register face (SPACE to capture, ESC to cancel) |
| **📱 Telegram** | Configure Telegram notifications |
| **🔒 Lock Now** | Lock screen immediately |
| **Quit** | Exit app |

### Register Face

1. Click **👤 Register Face**
2. Camera preview opens
3. Press **SPACE** to capture (or **ESC** to cancel)
4. Shows "Save successful!" when done

### Configure Telegram Notifications

1. Create bot: Chat with [@BotFather](https://t.me/BotFather) → `/newbot` → Get **Bot Token**
2. Get Chat ID: Chat with [@userinfobot](https://t.me/userinfobot) → Get **Chat ID**
3. In app: **📱 Telegram → Configure** → Enter Token and Chat ID
4. Toggle notifications: **Notify Unlock** / **Notify Lock**

### How it works

```
Face detected → 🔓 Unlocked
       ↓
No face (timeout seconds)
       ↓
🔒 Screen locked
       ↓
Registered face detected → Verify → Auto unlock
```

## Security

- **Face data encryption**: AES-128-CBC (Fernet) encryption
- **Key storage**: macOS Keychain (`com.vilock.face`)
- **Benefit**: Copying `data/` folder to another Mac won't work without Keychain access

## Data Storage

```
./data/
├── settings.plist              # Configuration
└── faces/
    ├── registered_face.enc     # Encrypted face image
    └── face_encoding.enc       # Encrypted face encoding
```

## Useful Commands

```bash
# Run app
python -m src.app

# Kill running app
pkill -f 'python.*src'

# Restart app
pkill -f 'python.*src'; sleep 1; python -m src.app

# View settings
/usr/libexec/PlistBuddy -c "Print" ./data/settings.plist
```

## Exit App

- Click **Quit** in menu bar
- Or press **Ctrl+C** in terminal

## License

MIT
