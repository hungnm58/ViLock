"""py2app setup for building AutoLock.app bundle."""

from setuptools import setup

APP = ['src/app.py']
DATA_FILES = ['assets']

OPTIONS = {
    'argv_emulation': False,
    'packages': ['cv2', 'mediapipe', 'rumps', 'numpy'],
    'includes': ['tkinter'],
    'excludes': ['matplotlib', 'scipy', 'pandas', 'PIL'],
    'resources': ['assets/'],
    'iconfile': 'assets/icon.icns',
    'plist': {
        'CFBundleName': 'AutoLock',
        'CFBundleDisplayName': 'AutoLock',
        'CFBundleIdentifier': 'com.autolock.app',
        'CFBundleVersion': '2.0.0',
        'CFBundleShortVersionString': '2.0.0',
        'LSUIElement': True,  # Menu bar only, no dock icon
        'NSHighResolutionCapable': True,
        'NSCameraUsageDescription': 'AutoLock uses the camera to detect your face and lock the screen when you leave.',
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
    }
}

setup(
    name='AutoLock',
    version='2.0.0',
    description='Auto lock screen when no face detected',
    author='hungnm',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
