"""Independent capture -> decision -> input -> verification loop."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time
import ctypes
import cv2
import numpy as np
from windows_driver import WindowsDriver,WindowMovedError,screen_point
from vision import observe,image_read,annotate,LABELS
from model import search,verify_transition
from exploration import Explorer,covered_hints
from occlusion import OcclusionBeliefs
from stability import stable
from configuration import load_assets


RESOURCE=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))
APP_HOME=Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent


def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def signature(state):
    return (tuple(sorted(state['tray']['items'])),state['tray']['complete'],
            tuple(sorted((t['kind'],round(t['click_point'][0]/3),round(t['click_point'][1]/3)) for t in state['tiles'])))


class Runner:
    def __init__(self,*,max_steps=120,max_seconds=300,explore=False,stop_event=None,
                 on_event=lambda e:None,session_dir=None,driver=None,config_path=None,template_dir=None):
        self.config,self.template_dir,self.templates=load_assets(config_path,template_dir)
        self.max_steps=max_steps;self.max_seconds=max_seconds;self.explore=explore
        self.stop_event=stop_event or threading.Event();self.on_event=on_event
        self.dir=Path(session_dir).resolve() if session_dir else APP_HOME/'standalone_sessions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.driver=driver;self.explorer=Explorer(reserve=2,max_explorations=6)
        self.occlusion=OcclusionBeliefs()
        self.clicked=0;self.verified=0;self.explored=0;self.cleared=0;self.pending=None
        self.state=None;self.mutex=None;self.last_window_rect=None

    def log(self,event,**data):
        entry={'time':datetime.now().isoformat(timespec='milliseconds'),'event':event,**data}
        with (self.dir/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        self.on_event(entry)

    def check_stop(self):
        if self.stop_event.is_set() or (self.dir/'STOP').exists() or self.driver.abort_pressed():
            raise InterruptedError('用户停止 / F8 / Esc')

    def wait(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.check_stop();self.stop_event.wait(min(.04,max(0,end-time.monotonic())))

    def settled(self,delay=.05):
        self.wait(delay)
        for _ in range(5):
            try:
                a=self.driver.capture();self.wait(.10);b=self.driver.capture()
                if a.rect!=b.rect:raise WindowMovedError('连续截图之间窗口移动。')
            except WindowMovedError as exc:
                self.log('reobserve',reason=str(exc));self.wait(.20);continue
            good,metrics=stable(a.image,b.image,self.config)
            if good:return b
            self.wait(.20)
        raise RuntimeError('画面连续不稳定，停止点击。')

    def record_location(self,rect=None):
        location=self.driver.location(rect)
        if location['rect']!=self.last_window_rect:
            previous=self.last_window_rect
            self.last_window_rect=location['rect']
            save_json(self.dir/'window_location.json',location)
            self.log('window_located',**location,previous_rect=previous)
        return location

    def recognize(self,frame):
        return observe(frame.image,self.config,self.template_dir)

    def acquire_mutex(self):
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p]
        kernel.CreateMutexW.restype=ctypes.c_void_p
        self.mutex=kernel.CreateMutexW(None,False,'Local\\SheepAutoExclusiveRunner')
        if not self.mutex:raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error()==183:
            kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle(self.mutex);self.mutex=None
            raise RuntimeError('已有一个独立程序正在控制游戏。')

    def run(self):
        if type(self.max_steps) is not int or self.max_steps<1 or not math.isfinite(self.max_seconds) or self.max_seconds<=0:
            raise ValueError('点击上限必须为正整数，时间上限必须为有限正数。')
        if self.dir.exists() and any(self.dir.iterdir()):raise RuntimeError('会话目录已存在，禁止覆盖。')
        self.dir.mkdir(parents=True,exist_ok=True)
        started=time.monotonic();reason='unknown';failed=False
        try:
            if self.driver is None and not self.config.get('ready_for_input',False):
                raise RuntimeError('尚未确认本机标定：先离线核对，再将 local/calibration.json 的 ready_for_input 设为 true。')
            self.acquire_mutex()
            self.log('locating_window',title=self.config['window_title'])
            if self.driver is None:self.driver=WindowsDriver(self.config['window_title'],self.config['frame_size'])
            self.driver.activate()
            self.record_location()
            self.log('started',runtime='native_win32_python',exploration=self.explore)
            while self.clicked<self.max_steps and time.monotonic()-started<self.max_seconds:
                self.check_stop()
                frame=self.settled();self.record_location(frame.rect)
                state=self.recognize(frame);self.state=state
                tag=f'{self.clicked+1:03}'
                folder=self.dir/tag;folder.mkdir(exist_ok=True)
                cv2.imencode('.png',frame.image)[1].tofile(str(folder/'before.png'))
                save_json(folder/'before.json',state)
                if not state['tray']['complete']:
                    reason=(f"槽位识别不完整：已识别 {len(state['tray']['items'])} 张，"
                            f"未识别区域 {len(state['tray']['unknown_regions'])} 处，"
                            f"候选冲突 {state['tray'].get('ambiguous_detections',0)} 个")
                    annotate(frame.image,state,folder/'recognition_issue.png')
                    save_json(folder/'decision.json',{'status':'pause','reason':reason})
                    self.log('recognition_incomplete',reason=reason,tray=state['tray'],
                             frame_path=str(folder/'recognition_issue.png'),state_path=str(folder/'before.json'))
                    break
                hints=covered_hints(frame.image,self.config,self.templates,state['tiles']) if self.explore else []
                belief=self.occlusion.infer(frame.image,state,hints,self.config)
                save_json(folder/'occlusion_probabilities.json',belief)
                plan=self.explorer.choose(state,hints,belief) if self.explore else search(state,time_limit=.35)
                save_json(folder/'hints.json',hints);save_json(folder/'decision.json',plan)
                rows=[]
                for node in belief['hypotheses']:
                    blockers='；'.join(f"{b['tile_id']} {LABELS.get(b['kind'],b['kind'])} {b['p_above_given_dark_tile']:.0%}"
                                      for b in node['potential_blockers'])
                    effects='；'.join(f"移除 {e['remove_tile_id']} → {e['p_exposed_after']:.0%}" for e in node['removal_effects'])
                    rows.append([node['id'],LABELS.get(node['kind_hypothesis'],'未知'),f"{node['p_real_dark_tile']:.0%}",blockers,effects])
                self.log('belief',rows=rows,calibrated=False,frame_path=str(folder/'before.png'))
                if plan['status']=='proposal' and plan.get('mode')=='explore':
                    self.log('exploration_decision',kind=plan['next_action']['kind'],
                             expected_exposure_utility=plan['next_action'].get('expected_exposure_utility',0),
                             reserved_slots=plan['next_action']['reserved_slots_after_action'])
                if plan['status']!='proposal':reason=plan.get('reason','无可执行决策');break
                action=plan['next_action'];action.setdefault('mode','known_clearance')
                # Recheck the actual scene after search. A new state is replanned,
                # never clicked through with stale coordinates.
                try:fresh=self.driver.capture()
                except WindowMovedError as exc:
                    self.log('reobserve',reason=str(exc));continue
                self.record_location(fresh.rect)
                if fresh.rect!=frame.rect:
                    self.log('reobserve',reason='决策期间窗口移动，重新识别后再决策');continue
                fresh_state=self.recognize(fresh)
                if signature(fresh_state)!=signature(state):
                    self.log('reobserve',reason='决策期间画面变化');self.wait(.15);continue
                target=next(t for t in state['tiles'] if t['id']==action['tile_id'])
                x,y,w,h=target['patch_bbox']
                patch_delta=np.abs(frame.image[y:y+h,x:x+w].astype(float)-fresh.image[y:y+h,x:x+w])
                if patch_delta.mean()>3:
                    self.log('reobserve',reason='目标牌面变化');continue
                self.check_stop()
                self.pending={'step':self.clicked+1,'action':action,'before':str(folder/'before.json'),
                              'window_rect':list(fresh.rect),
                              'desktop_point':list(screen_point(fresh.rect,self.driver.size,action['click_point']))}
                save_json(self.dir/'pending.json',self.pending)
                self.log('click_pending',**self.pending)
                try:self.driver.click(fresh,action['click_point'])
                except WindowMovedError as exc:
                    # This exception guarantees no input was sent. Only this
                    # case can discard a pending action and reobserve safely.
                    self.pending=None;save_json(self.dir/'pending.json',None)
                    self.log('click_cancelled_window_moved',reason=str(exc));continue
                self.clicked+=1
                if action['mode']=='explore':self.explored+=1
                self.log('clicked',step=self.clicked,kind=action['kind'],mode=action['mode'],
                         tray_before=state['tray']['items'],expected_tray=action['expected_tray_counts'])
                expected_len=sum(action['expected_tray_counts'].values())
                removes=len(state['tray']['items'])+1-expected_len
                after_frame=self.settled(1.0 if removes else .68)
                after=self.recognize(after_frame);self.state=after
                cv2.imencode('.png',after_frame.image)[1].tofile(str(folder/'after.png'))
                save_json(folder/'after.json',after);annotate(after_frame.image,after,folder/'annotated.png')
                ok,message=verify_transition(state,action,after)
                if not ok:
                    reason='操作核验失败：'+message;self.log('verification_failed',step=self.clicked,reason=reason);break
                self.verified+=1;self.cleared+=removes;self.pending=None
                save_json(self.dir/'pending.json',None)
                self.explorer.record(state,action,after)
                self.occlusion.update(belief,state,action,after)
                save_json(self.dir/'occlusion_learning.json',{'alpha':self.occlusion.alpha,'beta':self.occlusion.beta,
                          'outcomes':self.occlusion.evidence,'calibration_status':'uncalibrated; automatic weak labels'})
                self.log('verified',step=self.clicked,kind=action['kind'],mode=action['mode'],
                         tray=after['tray']['items'],cleared=removes,total_cleared=self.cleared,
                         newly_visible=self.explorer.history[-1]['newly_visible'],
                         frame_path=str(folder/'after.png'))
            else:
                reason='达到点击上限' if self.clicked>=self.max_steps else '达到时间上限'
        except InterruptedError as exc:reason=str(exc)
        except Exception as exc:
            failed=True
            reason=f'{type(exc).__name__}: {exc}'
            self.log('error',reason=reason)
        finally:
            if self.mutex:
                kernel=ctypes.WinDLL('kernel32');kernel.CloseHandle.argtypes=[ctypes.c_void_p]
                kernel.CloseHandle(self.mutex);self.mutex=None
        summary={'runtime':'standalone_native_windows','reason':reason,'failed':failed,'clicks':self.clicked,'verified':self.verified,
                 'exploration_clicks':self.explored,'cleared_tiles':self.cleared,'pending':self.pending is not None,
                 'tray':self.state['tray']['items'] if self.state else None,
                 'elapsed_seconds':round(time.monotonic()-started,2),'session_path':str(self.dir),
                 'level_clear_verified':False}
        save_json(self.dir/'summary.json',summary);self.log('finished',**summary)
        return summary
