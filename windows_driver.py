"""Standalone Win32 capture/input backend; no Codex, Node or network dependency."""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
import os
import time
import cv2
import numpy as np
from PIL import ImageGrab


@dataclass
class Frame:
    image: np.ndarray
    rect: tuple[int,int,int,int]
    captured_at: float


class WindowMovedError(RuntimeError):
    """Geometry changed before input; no button-down has been sent."""


def screen_point(rect,size,point):
    """Map a normalized image point using this frame's physical window bounds."""
    left,top,right,bottom=rect
    return (left+round(point[0]*(right-left)/size[0]),
            top+round(point[1]*(bottom-top)/size[1]))


if os.name=='nt':
    U=C.WinDLL('user32',use_last_error=True)
    D=C.WinDLL('dwmapi')
    U.GetWindowRect.argtypes=[W.HWND,C.POINTER(W.RECT)];U.GetWindowRect.restype=W.BOOL
    U.GetWindowTextW.argtypes=[W.HWND,W.LPWSTR,C.c_int]
    U.IsWindowVisible.argtypes=[W.HWND];U.IsWindowVisible.restype=W.BOOL
    U.IsWindow.argtypes=[W.HWND];U.IsWindow.restype=W.BOOL
    U.IsIconic.argtypes=[W.HWND];U.IsIconic.restype=W.BOOL
    U.GetForegroundWindow.restype=W.HWND
    U.GetWindowThreadProcessId.argtypes=[W.HWND,C.POINTER(W.DWORD)]
    U.SetForegroundWindow.argtypes=[W.HWND];U.SetForegroundWindow.restype=W.BOOL
    U.ShowWindow.argtypes=[W.HWND,C.c_int]
    U.WindowFromPoint.argtypes=[W.POINT];U.WindowFromPoint.restype=W.HWND
    U.GetAncestor.argtypes=[W.HWND,W.UINT];U.GetAncestor.restype=W.HWND
    U.SetCursorPos.argtypes=[C.c_int,C.c_int];U.SetCursorPos.restype=W.BOOL
    U.GetAsyncKeyState.argtypes=[C.c_int];U.GetAsyncKeyState.restype=C.c_short
    D.DwmGetWindowAttribute.argtypes=[W.HWND,W.DWORD,C.c_void_p,W.DWORD]
    CALLBACK=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
    U.EnumWindows.argtypes=[CALLBACK,W.LPARAM]

    class MOUSEINPUT(C.Structure):
        _fields_=[('dx',W.LONG),('dy',W.LONG),('mouseData',W.DWORD),('dwFlags',W.DWORD),
                  ('time',W.DWORD),('dwExtraInfo',C.c_size_t)]
    class KEYBDINPUT(C.Structure):
        _fields_=[('wVk',W.WORD),('wScan',W.WORD),('dwFlags',W.DWORD),('time',W.DWORD),('dwExtraInfo',C.c_size_t)]
    class HARDWAREINPUT(C.Structure):
        _fields_=[('uMsg',W.DWORD),('wParamL',W.WORD),('wParamH',W.WORD)]
    class INPUTUNION(C.Union):
        _fields_=[('mi',MOUSEINPUT),('ki',KEYBDINPUT),('hi',HARDWAREINPUT)]
    class INPUT(C.Structure):
        _anonymous_=['payload']
        _fields_=[('type',W.DWORD),('payload',INPUTUNION)]
    U.SendInput.argtypes=[W.UINT,C.POINTER(INPUT),C.c_int];U.SendInput.restype=W.UINT


def dpi_awareness():
    if os.name!='nt':raise RuntimeError('This desktop driver requires Windows.')
    try:
        U.SetProcessDpiAwarenessContext.argtypes=[C.c_void_p]
        U.SetProcessDpiAwarenessContext(C.c_void_p(-4))
    except AttributeError:
        try:C.WinDLL('shcore').SetProcessDpiAwareness(2)
        except OSError:pass
    # GUI workers also need physical pixels when the process was initialized by
    # another library before the driver was imported.
    if hasattr(U,'SetThreadDpiAwarenessContext'):
        U.SetThreadDpiAwarenessContext.argtypes=[C.c_void_p]
        U.SetThreadDpiAwarenessContext.restype=C.c_void_p
        U.SetThreadDpiAwarenessContext(C.c_void_p(-4))


def title(hwnd):
    buf=C.create_unicode_buffer(512);U.GetWindowTextW(hwnd,buf,len(buf));return buf.value


def window_rect(hwnd):
    r=W.RECT()
    if D.DwmGetWindowAttribute(hwnd,9,C.byref(r),C.sizeof(r))!=0:
        if not U.GetWindowRect(hwnd,C.byref(r)):raise C.WinError(C.get_last_error())
    return r.left,r.top,r.right,r.bottom


def windows():
    dpi_awareness();result=[]
    @CALLBACK
    def visit(hwnd,_):
        name=title(hwnd)
        if U.IsWindowVisible(hwnd) and name:
            result.append({'hwnd':int(hwnd),'title':name,'rect':window_rect(hwnd)})
        return True
    U.EnumWindows(visit,0)
    return result


class WindowsDriver:
    def __init__(self,window_title,expected_size):
        dpi_awareness()
        choices=[w for w in windows() if w['title']==window_title]
        if len(choices)!=1:raise RuntimeError(f'需唯一游戏窗口“{window_title}”，当前找到 {len(choices)} 个。')
        self.hwnd=choices[0]['hwnd'];self.title=window_title;self.size=tuple(expected_size)
        pid=W.DWORD();U.GetWindowThreadProcessId(self.hwnd,C.byref(pid));self.pid=pid.value

    @staticmethod
    def abort_pressed():
        return bool(U.GetAsyncKeyState(0x77)&0x8000 or U.GetAsyncKeyState(0x1B)&0x8000)

    def activate(self):
        if U.IsIconic(self.hwnd):U.ShowWindow(self.hwnd,9)
        U.SetForegroundWindow(self.hwnd)
        time.sleep(.12)
        self.validate()

    def validate(self,require_foreground=True):
        if not U.IsWindow(self.hwnd) or title(self.hwnd)!=self.title:raise RuntimeError('游戏窗口已关闭或发生变化。')
        pid=W.DWORD();U.GetWindowThreadProcessId(self.hwnd,C.byref(pid))
        if pid.value!=self.pid:raise RuntimeError('目标窗口进程已变化。')
        if U.IsIconic(self.hwnd):raise RuntimeError('游戏窗口被最小化。')
        if require_foreground and U.GetForegroundWindow()!=self.hwnd:raise RuntimeError('游戏失去前台焦点，已停止点击。')
        rect=window_rect(self.hwnd)
        sx=(rect[2]-rect[0])/self.size[0];sy=(rect[3]-rect[1])/self.size[1]
        if not (.5<=sx<=3 and .5<=sy<=3 and abs(sx/sy-1)<.007):
            raise RuntimeError(f'窗口纵横比与标定不符：{rect[2]-rect[0]}×{rect[3]-rect[1]}。')
        return rect

    def location(self,rect=None):
        """Read-only discovery result; never restores, moves or clicks a window."""
        if rect is None:rect=self.validate(require_foreground=False)
        left,top,right,bottom=rect
        return {'hwnd':self.hwnd,'pid':self.pid,'title':self.title,
                'rect':list(rect),'top_left':[left,top],
                'physical_size':[right-left,bottom-top],'normalized_size':list(self.size),
                'scale_xy':[(right-left)/self.size[0],(bottom-top)/self.size[1]],
                'coordinate_system':'physical desktop pixels; refreshed from the current HWND',
                'mapping':'desktop = current window top-left + normalized image point * current scale'}

    def owned(self,x,y):
        root=U.GetAncestor(U.WindowFromPoint(W.POINT(x,y)),2)
        return root==self.hwnd

    def capture(self):
        rect=self.validate();left,top,right,bottom=rect
        # Detect overlays at a grid of points, not just at the title bar.
        for fx in (.12,.5,.88):
            for fy in (.23,.45,.65,.82):
                if not self.owned(round(left+(right-left)*fx),round(top+(bottom-top)*fy)):
                    raise RuntimeError('游戏画面被其他窗口遮挡。')
        image=np.asarray(ImageGrab.grab(bbox=rect,all_screens=True,include_layered_windows=True).convert('RGB'))[:,:,::-1].copy()
        if self.validate()!=rect:raise WindowMovedError('截图时窗口移动，重新定位。')
        image=cv2.resize(image,self.size,interpolation=cv2.INTER_AREA)
        return Frame(image,rect,time.monotonic())

    def click(self,frame,point):
        if self.abort_pressed():raise InterruptedError('F8/Esc 急停。')
        if time.monotonic()-frame.captured_at>2:raise RuntimeError('点击所依据的截图过期。')
        if self.validate()!=frame.rect:raise WindowMovedError('窗口位置已变化，取消旧点击并重新定位。')
        x,y=point
        if not (0<x<self.size[0] and 52<y<self.size[1]-120):raise RuntimeError('点击点不在允许的牌面区域。')
        sx,sy=screen_point(frame.rect,self.size,point)
        if not self.owned(sx,sy):raise RuntimeError('目标点被其他窗口覆盖。')
        if not U.SetCursorPos(sx,sy):raise C.WinError(C.get_last_error())
        if self.validate()!=frame.rect:raise WindowMovedError('输入前窗口移动，本次尚未发送点击。')
        if not self.owned(sx,sy):raise RuntimeError('输入前目标点被其他窗口覆盖。')
        events=(INPUT*2)()
        events[0].type=0;events[0].mi=MOUSEINPUT(0,0,0,0x0002,0,0)
        events[1].type=0;events[1].mi=MOUSEINPUT(0,0,0,0x0004,0,0)
        sent=U.SendInput(2,events,C.sizeof(INPUT))
        if sent!=2:
            # Release only; never duplicate a button-down of unknown outcome.
            release=INPUT();release.type=0;release.mi=MOUSEINPUT(0,0,0,0x0004,0,0)
            U.SendInput(1,C.byref(release),C.sizeof(INPUT))
            raise RuntimeError(f'输入发送结果不完整（{sent}/2），保留待核验事务。')
        time.sleep(.04)
        # Park over empty green background to remove hover glow from the board.
        U.SetCursorPos(*screen_point(frame.rect,self.size,(self.size[0]-27,515)))
