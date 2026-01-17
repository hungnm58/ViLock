"""Preferences window using tkinter for native-like macOS experience."""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from .settings import load_settings, save_settings


class PreferencesWindow:
    """Tkinter preferences window for AutoLock settings."""

    def __init__(self, on_save: Optional[Callable] = None):
        """Initialize preferences window.

        Args:
            on_save: Callback function when settings are saved
        """
        self.on_save = on_save
        self.window: Optional[tk.Toplevel] = None
        self._root: Optional[tk.Tk] = None

    def show(self):
        """Show the preferences window."""
        if self.window is not None and self.window.winfo_exists():
            self.window.lift()
            self.window.focus_force()
            return

        # Create hidden root if needed (for standalone window)
        if self._root is None:
            self._root = tk.Tk()
            self._root.withdraw()

        self._create_window()

    def _create_window(self):
        """Create and configure the preferences window."""
        settings = load_settings()

        self.window = tk.Toplevel(self._root)
        self.window.title("AutoLock Preferences")
        self.window.geometry("380x320")
        self.window.resizable(False, False)
        self.window.attributes('-topmost', True)

        # Center on screen
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() - 380) // 2
        y = (self.window.winfo_screenheight() - 320) // 3
        self.window.geometry(f"+{x}+{y}")

        # Main frame with padding
        main = ttk.Frame(self.window, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        # Timeout setting
        ttk.Label(main, text="Lock timeout (seconds):").pack(anchor=tk.W)
        self.timeout_var = tk.IntVar(value=settings['timeout'])
        timeout_frame = ttk.Frame(main)
        timeout_frame.pack(fill=tk.X, pady=(5, 15))

        self.timeout_scale = ttk.Scale(
            timeout_frame, from_=5, to=60,
            variable=self.timeout_var, orient=tk.HORIZONTAL
        )
        self.timeout_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.timeout_label = ttk.Label(timeout_frame, text=f"{settings['timeout']}s")
        self.timeout_label.pack(side=tk.LEFT, padx=(10, 0))
        self.timeout_scale.configure(command=self._update_timeout_label)

        # Confidence threshold
        ttk.Label(main, text="Detection sensitivity:").pack(anchor=tk.W)
        self.confidence_var = tk.DoubleVar(value=settings['confidence_threshold'])
        conf_frame = ttk.Frame(main)
        conf_frame.pack(fill=tk.X, pady=(5, 15))

        self.conf_scale = ttk.Scale(
            conf_frame, from_=0.3, to=0.9,
            variable=self.confidence_var, orient=tk.HORIZONTAL
        )
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        conf_pct = int(settings['confidence_threshold'] * 100)
        self.conf_label = ttk.Label(conf_frame, text=f"{conf_pct}%")
        self.conf_label.pack(side=tk.LEFT, padx=(10, 0))
        self.conf_scale.configure(command=self._update_conf_label)

        # Camera index
        ttk.Label(main, text="Camera index:").pack(anchor=tk.W)
        self.camera_var = tk.IntVar(value=settings['camera_index'])
        camera_frame = ttk.Frame(main)
        camera_frame.pack(fill=tk.X, pady=(5, 15))

        for i in range(3):
            ttk.Radiobutton(
                camera_frame, text=f"Camera {i}",
                variable=self.camera_var, value=i
            ).pack(side=tk.LEFT, padx=(0, 15))

        # Start at login
        self.autostart_var = tk.BooleanVar(value=settings['start_at_login'])
        ttk.Checkbutton(
            main, text="Start at login",
            variable=self.autostart_var
        ).pack(anchor=tk.W, pady=(0, 20))

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Cancel", command=self.window.destroy).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side=tk.RIGHT)

    def _update_timeout_label(self, value):
        """Update timeout label when slider changes."""
        self.timeout_label.configure(text=f"{int(float(value))}s")

    def _update_conf_label(self, value):
        """Update confidence label when slider changes."""
        self.conf_label.configure(text=f"{int(float(value) * 100)}%")

    def _save(self):
        """Save settings and close window."""
        settings = {
            'timeout': int(self.timeout_var.get()),
            'confidence_threshold': round(self.confidence_var.get(), 2),
            'camera_index': self.camera_var.get(),
            'start_at_login': self.autostart_var.get(),
        }

        current = load_settings()
        current.update(settings)
        save_settings(current)

        if self.on_save:
            self.on_save(current)

        self.window.destroy()

    def close(self):
        """Close preferences window if open."""
        if self.window and self.window.winfo_exists():
            self.window.destroy()
        if self._root:
            self._root.destroy()
            self._root = None
