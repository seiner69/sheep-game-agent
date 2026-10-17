"""Coordinate relocation and cancellation before a mouse-down."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import windows_driver as wd
from standalone_runner import Runner
from synthetic import write_fixture


class WindowLocationTests(unittest.TestCase):
    def setUp(self):
        self.assets=tempfile.TemporaryDirectory()
        self.addCleanup(self.assets.cleanup)
        self.config_path,self.template_dir=write_fixture(Path(self.assets.name)/'assets')

    def test_relocation_changes_only_the_desktop_offset(self):
        old=(109,172,666,1235);new=(409,222,966,1285);size=(371,709)
        point=(205.5,237.0)
        before=wd.screen_point(old,size,point);after=wd.screen_point(new,size,point)
        self.assertEqual((after[0]-before[0],after[1]-before[1]),(300,50))

    def test_negative_monitor_origin_and_scaling(self):
        self.assertEqual(wd.screen_point((-1000,50,-258,1468),(371,709),(100,200)),(-800,450))

    def test_every_new_driver_discovers_current_window_handle(self):
        with patch.object(wd,'dpi_awareness'),patch.object(wd,'U',create=True),patch.object(wd,'windows') as windows:
            windows.side_effect=[[{'title':'game','hwnd':100}],[{'title':'game','hwnd':200}]]
            self.assertEqual(wd.WindowsDriver('game',(371,709)).hwnd,100)
            self.assertEqual(wd.WindowsDriver('game',(371,709)).hwnd,200)
            self.assertEqual(windows.call_count,2)

    def test_missing_or_ambiguous_windows_are_not_selected(self):
        with patch.object(wd,'dpi_awareness'),patch.object(wd,'windows') as windows:
            for choices in ([],[{'title':'game','hwnd':1},{'title':'game','hwnd':2}]):
                windows.return_value=choices
                with self.assertRaises(RuntimeError):wd.WindowsDriver('game',(371,709))

    def driver(self):
        d=object.__new__(wd.WindowsDriver);d.size=(371,709)
        d.abort_pressed=lambda:False;d.owned=lambda *args:True
        return d

    def test_move_before_click_does_not_even_move_the_pointer(self):
        d=self.driver();d.validate=lambda:(200,100,757,1163)
        frame=wd.Frame(np.zeros((709,371,3),np.uint8),(109,172,666,1235),time.monotonic())
        with patch.object(wd,'U',create=True) as win:
            with self.assertRaises(wd.WindowMovedError):d.click(frame,(100,200))
            win.SetCursorPos.assert_not_called();win.SendInput.assert_not_called()

    def test_move_after_pointer_positioning_still_cancels_button_down(self):
        d=self.driver();old=(109,172,666,1235)
        d.validate=Mock(side_effect=[old,(200,100,757,1163)])
        frame=wd.Frame(np.zeros((709,371,3),np.uint8),old,time.monotonic())
        with patch.object(wd,'U',create=True) as win:
            win.SetCursorPos.return_value=True
            with self.assertRaises(wd.WindowMovedError):d.click(frame,(100,200))
            win.SetCursorPos.assert_called_once();win.SendInput.assert_not_called()

    def test_runner_clears_only_a_definitely_cancelled_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            rect=(109,172,666,1235);frame=wd.Frame(np.zeros((709,371,3),np.uint8),rect,time.monotonic())
            d=Mock();d.size=(371,709);d.capture.return_value=frame
            d.location.return_value={'rect':list(rect)};d.abort_pressed.return_value=False
            def move(*args):
                d.abort_pressed.return_value=True
                raise wd.WindowMovedError('moved before input')
            d.click.side_effect=move
            runner=Runner(max_steps=1,explore=True,session_dir=Path(tmp)/'run',driver=d,config_path=self.config_path,template_dir=self.template_dir)
            runner.acquire_mutex=lambda:None;runner.settled=lambda *args:frame
            state={'frame_id':'one','tray':{'items':[],'complete':True},
                   'tiles':[{'id':'T1','kind':'a','click_point':[100,200],'patch_bbox':[0,0,3,3]}]}
            runner.recognize=lambda f:state
            runner.explorer.choose=lambda *args:{'status':'proposal','next_action':{
                'tile_id':'T1','kind':'a','click_point':[100,200],'expected_tray_counts':{'a':1}}}
            runner.occlusion.infer=lambda *args:{'hypotheses':[]}
            with patch('standalone_runner.covered_hints',return_value=[]):result=runner.run()
            self.assertFalse(result['pending']);self.assertEqual(result['clicks'],0)
            self.assertIsNone(json.loads((Path(tmp)/'run/pending.json').read_text()))
            self.assertIn('click_cancelled_window_moved',(Path(tmp)/'run/events.jsonl').read_text(encoding='utf-8'))

    def test_initial_incomplete_tray_saves_diagnostics_without_clicking(self):
        with tempfile.TemporaryDirectory() as tmp:
            rect=(109,172,666,1235);frame=wd.Frame(np.zeros((709,371,3),np.uint8),rect,time.monotonic())
            d=Mock();d.location.return_value={'rect':list(rect)};d.abort_pressed.return_value=False
            runner=Runner(max_steps=1,session_dir=Path(tmp)/'run',driver=d,config_path=self.config_path,template_dir=self.template_dir)
            runner.acquire_mutex=lambda:None;runner.settled=lambda *args:frame
            runner.recognize=lambda f:{'tray':{'items':['fire'],'complete':False,'unknown_regions':[[40,1,44,45]]},
                                      'tiles':[],'covered_observations':[]}
            result=runner.run()
            self.assertEqual(result['clicks'],0);d.click.assert_not_called()
            self.assertTrue((Path(tmp)/'run/001/before.json').exists())
            self.assertTrue((Path(tmp)/'run/001/recognition_issue.png').exists())


if __name__=='__main__':unittest.main()
