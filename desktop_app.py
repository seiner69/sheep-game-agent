"""Double-click GUI or --headless; fully offline and independent of Codex."""
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
from standalone_runner import Runner,RESOURCE,APP_HOME
from windows_driver import dpi_awareness,WindowsDriver
from configuration import load_assets


def gui(config_path=None,template_dir=None):
    import tkinter as tk
    from tkinter import ttk,messagebox
    from PIL import Image,ImageTk
    dpi_awareness()
    root=tk.Tk();root.title('羊了个羊 · 独立自动程序');root.geometry('980x760')
    style=ttk.Style();style.theme_use('clam');style.configure('TButton',padding=8)
    events=queue.Queue();stop=threading.Event();worker=None;closing=False
    top=ttk.Frame(root,padding=16);top.pack(fill='x')
    ttk.Label(top,text='定位窗口 → 识别 → 决策 → 点击 → 核验',font=('Microsoft YaHei UI',15,'bold')).pack(anchor='w')
    ttk.Label(top,text='完全本地运行   ·   F8 或 Esc 急停   ·   游戏失去焦点即停止').pack(anchor='w',pady=(8,0))
    options=ttk.Frame(root,padding=(16,0));options.pack(fill='x')
    ttk.Label(options,text='最多点击').pack(side='left');steps=tk.IntVar(value=1)
    ttk.Spinbox(options,from_=1,to=1000,textvariable=steps,width=6).pack(side='left',padx=8)
    explore=tk.BooleanVar(value=False);ttk.Checkbutton(options,text='允许探索隐藏牌（保留两格）',variable=explore).pack(side='left')
    status=tk.StringVar(value='准备就绪：先打开第二关，再按开始。')
    ttk.Label(root,textvariable=status,padding=16,wraplength=710).pack(fill='x')
    window_status=tk.StringVar(value='窗口尚未定位；开始时会自动查找，也可先点击“定位窗口”。')
    ttk.Label(root,textvariable=window_status,padding=(16,0,16,10),wraplength=930).pack(fill='x')
    # Reserve controls before the expanding content, including at high DPI.
    controls=ttk.Frame(root,padding=16);controls.pack(side='bottom',fill='x')
    body=ttk.Frame(root,padding=(16,0,16,16));body.pack(fill='both',expand=True)
    preview=ttk.Label(body);preview.pack(side='right',padx=(12,0))
    notebook=ttk.Notebook(body);notebook.pack(side='left',fill='both',expand=True)
    log=tk.Text(notebook,width=43,height=12,wrap='word',font=('Microsoft YaHei UI',10));notebook.add(log,text='运行日志')
    beliefs=ttk.Frame(notebook);notebook.add(beliefs,text='暗牌遮挡估计')
    ttk.Label(beliefs,text='几何先验估计，未经统计校准；遮挡概率以该暗牌确实存在为条件。',wraplength=600).pack(fill='x',pady=6)
    tree=ttk.Treeview(beliefs,columns=('id','kind','exists','blockers','reveal'),show='headings')
    for key,name,width in [('id','位置',88),('kind','疑似图案',70),('exists','存在估计',70),('blockers','可能的上层牌',180),('reveal','移除后露出估计',180)]:
        tree.heading(key,text=name);tree.column(key,width=width,minwidth=55,stretch=True)
    tree.pack(fill='both',expand=True)
    current_dir=tk.StringVar(value='')

    def append(text):
        log.insert('end',text+'\n');log.see('end')
    def locate():
        if worker and worker.is_alive():return
        try:
            config,_,_=load_assets(config_path,template_dir)
            location=WindowsDriver(config['window_title'],config['frame_size']).location()
            events.put({'event':'window_located',**location})
        except Exception as exc:
            window_status.set('定位失败：'+str(exc));append(window_status.get())
    def start():
        nonlocal worker
        if worker and worker.is_alive():return
        try:limit=steps.get();assert limit>0
        except Exception:messagebox.showerror('运行上限','请输入正整数。');return
        try:
            config,_,_=load_assets(config_path,template_dir)
            if not config.get('ready_for_input',False):raise ValueError('请先离线核对本机标定，再确认 ready_for_input。')
        except (ValueError,OSError) as exc:messagebox.showerror('标定未就绪',str(exc));return
        if not messagebox.askyesno('真实鼠标输入','将激活目标窗口并控制鼠标。确认标定与模板属于当前界面，且允许此次操作？'):return
        stop.clear();start_button.configure(state='disabled');status.set('正在绑定游戏窗口…')
        # Read Tk state on the UI thread; the worker uses the immutable value.
        allow=explore.get()
        def work_safe():
            try:Runner(max_steps=limit,max_seconds=max(180,limit*4),explore=allow,stop_event=stop,on_event=events.put,config_path=config_path,template_dir=template_dir).run()
            except Exception as exc:events.put({'event':'finished','reason':str(exc),'clicks':0,'verified':0})
        worker=threading.Thread(target=work_safe,daemon=True);worker.start()
    def stop_run():stop.set();status.set('正在停止；不会发送新的点击。')
    ttk.Button(controls,text='定位窗口',command=locate).pack(side='left',padx=(0,10))
    start_button=ttk.Button(controls,text='开始自动运行',command=start);start_button.pack(side='left')
    ttk.Button(controls,text='停止',command=stop_run).pack(side='left',padx=10)
    ttk.Label(controls,text='当前适配：羊了个羊：星球 / 第二关').pack(side='right')
    def poll():
        nonlocal closing
        while True:
            try:e=events.get_nowait()
            except queue.Empty:break
            if e['event']=='locating_window':
                window_status.set('正在查找游戏窗口，读取当前位置与尺寸…')
            elif e['event']=='window_located':
                x,y=e['top_left'];w,h=e['physical_size']
                window_status.set(f"已定位：左上角 ({x}, {y}) · 大小 {w} × {h} 像素 · 每次截图/点击重新检查")
                append(window_status.get())
            elif e['event'] in ('reobserve','click_cancelled_window_moved'):
                append(e['reason'])
            elif e['event']=='board_scale_fitted':
                append(f"牌面尺寸已适配：模板缩放 {e['scale']:.3f}，识别明牌 {e['open_tiles']} 张。")
            elif e['event']=='recognition_incomplete':
                append(e['reason']);append('诊断截图：'+e['frame_path'])
                try:
                    im=Image.open(e['frame_path']);im.thumbnail((230,440))
                    photo=ImageTk.PhotoImage(im);preview.configure(image=photo);preview.image=photo
                except Exception:pass
            elif e['event']=='belief':
                tree.delete(*tree.get_children())
                for row in e['rows']:tree.insert('','end',values=row)
            elif e['event']=='exploration_decision':
                append(f"探索依据：预计揭露效用 {e['expected_exposure_utility']:.3f}，保留 {e['reserved_slots']} 格。")
            elif e['event']=='verified':
                mode='探索' if e['mode']=='explore' else '消除搜索'
                status.set(f"第 {e['step']} 步已核验 · {mode} · 累计消除 {e['total_cleared']} 张 · 槽位 {len(e['tray'])}/7")
                append(f"#{e['step']} {mode}：{e['kind']}\n槽位：{', '.join(e['tray']) or '空'}")
                try:
                    im=Image.open(e['frame_path']);im.thumbnail((230,440))
                    photo=ImageTk.PhotoImage(im);preview.configure(image=photo);preview.image=photo
                except Exception:pass
            elif e['event']=='finished':
                status.set(f"已停止：{e['reason']} · {e.get('verified',0)}/{e.get('clicks',0)} 步通过核验")
                append(status.get());append('日志：'+e.get('session_path',''))
                current_dir.set(e.get('session_path',''));start_button.configure(state='normal')
        if closing and (not worker or not worker.is_alive()):root.destroy();return
        root.after(100,poll)
    def close():
        nonlocal closing
        closing=True;stop.set()
    root.protocol('WM_DELETE_WINDOW',close);root.bind('<F8>',lambda e:stop_run());root.bind('<Escape>',lambda e:stop_run())
    root.after(100,poll);root.mainloop()


def main():
    p=argparse.ArgumentParser(description='局部可见三消识别、决策和 Windows 执行；不附带游戏素材。')
    p.add_argument('--config',type=Path,default=RESOURCE/'local/calibration.json')
    p.add_argument('--templates',type=Path,default=RESOURCE/'local/templates')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--allow-input',action='store_true',help='Explicitly allow real mouse input in headless mode.')
    p.add_argument('--max-steps',type=int,default=1)
    p.add_argument('--max-seconds',type=float,default=60)
    p.add_argument('--explore',action='store_true',help='Opt into risk-bounded hidden-card exploration.')
    p.add_argument('--session-dir',type=Path)
    p.add_argument('--self-check',action='store_true')
    p.add_argument('--locate-only',action='store_true',help='Locate the configured window without activating or clicking it.')
    args=p.parse_args()
    try:
        config,_,templates=load_assets(args.config,args.templates)
    except (ValueError,OSError,KeyError) as exc:
        p.error(str(exc))
    if args.self_check:
        print(json.dumps({'ok':True,'template_files':len(templates),
                          'ready_for_input':config.get('ready_for_input',False),
                          'scope':'configuration and asset loading only; recognition accuracy not verified'},
                         ensure_ascii=False))
    elif args.locate_only:
        result=WindowsDriver(config['window_title'],config['frame_size']).location()
        print(json.dumps(result,ensure_ascii=False))
    elif args.headless:
        if not args.allow_input:p.error('--headless requires --allow-input; use demo.py for an offline run.')
        if not config.get('ready_for_input',False):p.error('ready_for_input is false; verify your local calibration first.')
        def report(event):
            if event['event'] in ('verified','finished','error'):
                print(json.dumps(event,ensure_ascii=False),flush=True)
        result=Runner(max_steps=args.max_steps,max_seconds=args.max_seconds,explore=args.explore,
                      session_dir=args.session_dir,on_event=report,
                      config_path=args.config,template_dir=args.templates).run()
        if result['failed'] or result['pending'] or result['verified']!=result['clicks']:
            raise SystemExit(1)
    else:
        gui(args.config,args.templates)


if __name__=='__main__':main()
