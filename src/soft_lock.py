"""Soft lock overlay window - fullscreen black window that dismisses on face detection."""

import threading
from typing import Callable, Optional

try:
    import AppKit
    from PyObjCTools import AppHelper
    APPKIT_AVAILABLE = True
except ImportError:
    APPKIT_AVAILABLE = False


class SoftLockWindow:
    """Fullscreen black overlay for all screens."""

    def __init__(self, on_unlock: Optional[Callable] = None):
        if not APPKIT_AVAILABLE:
            raise ImportError("PyObjC (AppKit) required for soft lock")

        self.on_unlock = on_unlock
        self.windows: list = []  # Windows for all screens
        self.is_locked = False

    def lock(self):
        """Show fullscreen black overlay on ALL screens."""
        if self.is_locked:
            return

        self.is_locked = True
        AppHelper.callAfter(self._create_windows)

    def _create_windows(self):
        """Create fullscreen black windows on ALL screens."""
        if self.windows:
            return

        # Get all screens
        screens = AppKit.NSScreen.screens()

        for i, screen in enumerate(screens):
            frame = screen.frame()

            # Create borderless window for this screen
            window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                AppKit.NSWindowStyleMaskBorderless,
                AppKit.NSBackingStoreBuffered,
                False
            )

            # Configure window
            window.setBackgroundColor_(AppKit.NSColor.blackColor())
            window.setOpaque_(True)
            window.setLevel_(AppKit.NSScreenSaverWindowLevel)
            window.setCollectionBehavior_(
                AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces |
                AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
            )

            # Add content only on main screen (first screen)
            if i == 0:
                content_view = window.contentView()

                # "Locked" text
                text_field = AppKit.NSTextField.alloc().initWithFrame_(
                    AppKit.NSMakeRect(frame.size.width / 2 - 150, frame.size.height / 2 - 30, 300, 60)
                )
                text_field.setStringValue_("🔒 Screen Locked")
                text_field.setFont_(AppKit.NSFont.systemFontOfSize_(36))
                text_field.setTextColor_(AppKit.NSColor.whiteColor())
                text_field.setBackgroundColor_(AppKit.NSColor.clearColor())
                text_field.setBezeled_(False)
                text_field.setEditable_(False)
                text_field.setAlignment_(AppKit.NSTextAlignmentCenter)
                content_view.addSubview_(text_field)

                # Subtitle
                subtitle = AppKit.NSTextField.alloc().initWithFrame_(
                    AppKit.NSMakeRect(frame.size.width / 2 - 200, frame.size.height / 2 - 80, 400, 30)
                )
                subtitle.setStringValue_("Face detection active - show your face to unlock")
                subtitle.setFont_(AppKit.NSFont.systemFontOfSize_(14))
                subtitle.setTextColor_(AppKit.NSColor.grayColor())
                subtitle.setBackgroundColor_(AppKit.NSColor.clearColor())
                subtitle.setBezeled_(False)
                subtitle.setEditable_(False)
                subtitle.setAlignment_(AppKit.NSTextAlignmentCenter)
                content_view.addSubview_(subtitle)

            # Show window
            window.makeKeyAndOrderFront_(None)
            window.orderFrontRegardless()

            self.windows.append(window)

    def unlock(self):
        """Hide all overlay windows."""
        if not self.is_locked:
            return

        self.is_locked = False
        AppHelper.callAfter(self._close_windows)

        # Call callback after unlock
        if self.on_unlock:
            try:
                self.on_unlock()
            except Exception:
                pass

    def _close_windows(self):
        """Close all windows on main thread."""
        for window in self.windows:
            window.orderOut_(None)
        self.windows = []

    def toggle(self):
        """Toggle lock state."""
        if self.is_locked:
            self.unlock()
        else:
            self.lock()
