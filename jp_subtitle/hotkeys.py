"""다른 앱을 보고 있을 때도 쓸 수 있는 전역 단축키 (Windows, macOS)."""

from __future__ import annotations

import ctypes
import sys
import traceback
from typing import Callable

from PySide6.QtCore import QAbstractNativeEventFilter, QCoreApplication

WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000

# Windows 가상 키 코드
HOTKEY_KEYS = {"R": 0x52, "Q": 0x51, "S": 0x53}
# macOS 키 코드 (Carbon)
MAC_KEY_CODES = {"R": 15, "Q": 12, "S": 1}
MAC_CONTROL_KEY = 0x1000
MAC_OPTION_KEY = 0x0800


def hotkey_name(key: str) -> str:
    """화면에 보여 줄 단축키 이름. 맥은 ⌃(Control) ⌥(Option) 기호로 쓴다."""
    return f"⌃⌥{key}" if sys.platform == "darwin" else f"Ctrl+Alt+{key}"


def _fourcc(text: str) -> int:
    return int.from_bytes(text.encode("ascii"), "big")


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _HotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_MAC_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
_MAC_SIGNATURE = _fourcc("JPSB")
_MAC_KEYBOARD_CLASS = _fourcc("keyb")
_MAC_HOTKEY_PRESSED = 5
_MAC_DIRECT_OBJECT = _fourcc("----")
_MAC_HOTKEY_ID_TYPE = _fourcc("hkid")


class _MacHotkeys:
    """Carbon 의 전역 단축키 API. 손쉬운 사용 권한 없이 동작한다."""

    def __init__(self, on_hotkey: Callable[[int], None], carbon=None):
        self._on_hotkey = on_hotkey
        self._carbon = carbon or ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
        c = self._carbon
        c.GetApplicationEventTarget.restype = ctypes.c_void_p
        c.GetApplicationEventTarget.argtypes = []
        c.InstallEventHandler.restype = ctypes.c_int32
        c.InstallEventHandler.argtypes = [
            ctypes.c_void_p, _MAC_HANDLER, ctypes.c_ulong,
            ctypes.POINTER(_EventTypeSpec), ctypes.c_void_p, ctypes.c_void_p,
        ]
        c.GetEventParameter.restype = ctypes.c_int32
        c.GetEventParameter.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p,
        ]
        c.RegisterEventHotKey.restype = ctypes.c_int32
        c.RegisterEventHotKey.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, _HotKeyID, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
        ]
        c.UnregisterEventHotKey.restype = ctypes.c_int32
        c.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
        self._target = c.GetApplicationEventTarget()
        self._handler = _MAC_HANDLER(self._handle)  # 지워지지 않게 붙들어 둔다
        self._refs: dict[int, ctypes.c_void_p] = {}
        spec = _EventTypeSpec(_MAC_KEYBOARD_CLASS, _MAC_HOTKEY_PRESSED)
        status = c.InstallEventHandler(
            self._target, self._handler, 1, ctypes.byref(spec), None, None
        )
        if status != 0:
            raise OSError(f"InstallEventHandler 실패: {status}")

    def _handle(self, _call, event, _user) -> int:
        try:
            hot_id = _HotKeyID()
            status = self._carbon.GetEventParameter(
                event, _MAC_DIRECT_OBJECT, _MAC_HOTKEY_ID_TYPE, None,
                ctypes.sizeof(hot_id), None, ctypes.byref(hot_id),
            )
            if status == 0 and hot_id.signature == _MAC_SIGNATURE:
                self._on_hotkey(hot_id.id)
        except Exception:
            traceback.print_exc()  # 단축키 때문에 프로그램이 죽으면 안 된다
        return 0

    def register(self, hotkey_id: int, key: str) -> bool:
        ref = ctypes.c_void_p()
        status = self._carbon.RegisterEventHotKey(
            MAC_KEY_CODES[key],
            MAC_CONTROL_KEY | MAC_OPTION_KEY,
            _HotKeyID(_MAC_SIGNATURE, hotkey_id),
            self._target,
            0,
            ctypes.byref(ref),
        )
        if status != 0:
            return False
        self._refs[hotkey_id] = ref
        return True

    def unregister_all(self) -> None:
        for ref in self._refs.values():
            self._carbon.UnregisterEventHotKey(ref)
        self._refs.clear()


class GlobalHotkeys(QAbstractNativeEventFilter):
    """Ctrl+Alt+<키> (맥은 ⌃⌥<키>) 를 등록한다. 등록하지 못한 키는 failed 에 남긴다."""

    def __init__(self) -> None:
        super().__init__()
        self._callbacks: dict[int, Callable[[], None]] = {}
        self.failed: list[str] = []
        self._mac: _MacHotkeys | None = None
        self.available = sys.platform == "win32"
        if self.available:
            QCoreApplication.instance().installNativeEventFilter(self)
        elif sys.platform == "darwin":
            try:
                self._mac = _MacHotkeys(self._on_mac_hotkey)
                self.available = True
            except Exception as e:
                print(f"[단축키] 맥 전역 단축키를 쓸 수 없습니다: {e}")

    def _on_mac_hotkey(self, hotkey_id: int) -> None:
        callback = self._callbacks.get(hotkey_id)
        if callback is not None:
            callback()

    def register(self, key: str, callback: Callable[[], None]) -> None:
        if not self.available:
            return
        hotkey_id = len(self._callbacks) + 1
        if self._mac is not None:
            ok = self._mac.register(hotkey_id, key)
        else:
            ok = ctypes.windll.user32.RegisterHotKey(
                None, hotkey_id, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, HOTKEY_KEYS[key]
            )
        if ok:
            self._callbacks[hotkey_id] = callback
        else:
            self.failed.append(hotkey_name(key))

    def unregister_all(self) -> None:
        if not self.available:
            return
        if self._mac is not None:
            self._mac.unregister_all()
        else:
            for hotkey_id in self._callbacks:
                ctypes.windll.user32.UnregisterHotKey(None, hotkey_id)
        self._callbacks.clear()

    def nativeEventFilter(self, event_type, message):
        if bytes(event_type) == b"windows_generic_MSG":
            from ctypes import wintypes

            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY and msg.wParam in self._callbacks:
                self._callbacks[msg.wParam]()
                return True, 0
        return False, 0
