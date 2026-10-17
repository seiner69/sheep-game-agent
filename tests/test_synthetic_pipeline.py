import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from configuration import load_assets, validate_config
from demo import run_demo
from model import search
from standalone_runner import Runner
from stability import stable
from synthetic import config, frame, template, write_fixture
from vision import detect, observe
from windows_driver import Frame, WindowsDriver


@pytest.fixture
def assets(tmp_path):
    return write_fixture(tmp_path / 'assets')


def test_demo_recognizes_plans_and_verifies_three_steps():
    result = run_demo()
    assert result['verified_steps'] == 3
    assert result['final_tray'] == []
    assert result['remaining_visible_tiles'] == 0
    assert result['real_input_sent'] is False
    assert result['game_clear_claim'] is False


def test_fixtures_keep_hidden_inventory_unknown(assets):
    cfg, directory, _ = load_assets(*assets)
    state = observe(frame(), cfg, directory)
    assert len(state['tiles']) == 3
    assert state['tray']['complete']
    assert state['knowledge']['remaining_total'] is None
    assert not state['occlusion_graph_complete']


def test_occupied_synthetic_trays(assets):
    cfg, directory, _ = load_assets(*assets)
    for count in range(4):
        state = observe(frame(count), cfg, directory)
        assert state['tray']['complete']
        assert state['tray']['items'] == ['synthetic_a'] * (count % 3)


def test_darkened_template_not_clickable():
    tile = template()
    hits, _ = detect((tile * .45).astype(np.uint8), [0, 0, 28, 28], {'a': tile})
    assert hits == []


def test_conflicting_template_labels_abstain():
    hits, ambiguous = detect(template(), [0, 0, 28, 28], {'a': template(), 'b': template()})
    assert not hits and ambiguous == 2


def test_unknown_tray_is_not_empty(assets):
    cfg, directory, _ = load_assets(*assets)
    image = frame()
    image[565:590, 250:275] = 230
    state = observe(image, cfg, directory)
    assert not state['tray']['complete']
    assert search(state)['status'] == 'pause'


def test_frame_size_and_annotations_are_not_reused(assets):
    cfg, directory, _ = load_assets(*assets)
    cfg['sample_frame_id'] = 'unrelated'
    cfg['sample_annotations'] = {'covered_observations': [{'id': 'not-current'}]}
    assert observe(frame(), cfg, directory)['covered_observations'] == []
    with pytest.raises(ValueError):
        observe(frame()[:-1], cfg, directory)


@pytest.mark.parametrize('field,value', [
    ('frame_size', [0, 709]), ('board_roi', [319, 140, 52, 487]),
    ('tray_roi', [22, 556, 900, 607]), ('minimum_score', float('nan')),
    ('maximum_mae', -1), ('ready_for_input', 'yes'), ('tray_scales', []),
    ('rules', {'tray_capacity': 9, 'match_size': 3}), ('window_title', ''),
])
def test_bad_calibration_is_rejected(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_overlapping_regions_rejected():
    cfg = config()
    cfg['tray_roi'] = cfg['board_roi']
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_missing_assets_and_blank_template_rejected(assets, tmp_path):
    with pytest.raises(ValueError):
        load_assets(tmp_path / 'missing.json', tmp_path / 'missing')
    cfg, directory = assets
    (directory / 'synthetic_a.png').unlink()
    with pytest.raises(ValueError):
        load_assets(cfg, directory)
    cv2.imencode('.png', np.zeros((28, 28, 3), np.uint8))[1].tofile(str(directory / 'blank.png'))
    with pytest.raises(ValueError):
        load_assets(cfg, directory)


def test_stability_checks_motion_and_shape():
    a = frame()
    assert stable(a, a.copy(), config())[0]
    assert not stable(a, a[:-1], config())[0]
    b = a.copy()
    b[140:487, 52:319] = 255
    assert not stable(a, b, config())[0]


class SyntheticDriver:
    size = (371, 709)
    rect = (0, 0, 371, 709)

    def __init__(self):
        self.clicks = 0

    def activate(self):
        pass

    def abort_pressed(self):
        return False

    def location(self, rect=None):
        return {'rect': list(self.rect), 'top_left': [0, 0], 'physical_size': list(self.size)}

    def capture(self):
        return Frame(frame(self.clicks), self.rect, time.monotonic())

    def click(self, snapshot, point):
        assert snapshot.rect == self.rect
        assert point == [100 + 60*self.clicks, 200]
        self.clicks += 1


def test_runner_full_loop_without_real_input(assets, tmp_path):
    driver = SyntheticDriver()
    runner = Runner(max_steps=3, explore=False, session_dir=tmp_path/'run', driver=driver,
                    config_path=assets[0], template_dir=assets[1])
    runner.acquire_mutex = lambda: None
    runner.wait = lambda seconds: None
    with patch.object(WindowsDriver, 'click', side_effect=AssertionError('Real input forbidden')):
        result = runner.run()
    assert result['clicks'] == result['verified'] == 3
    assert not result['pending'] and result['tray'] == []
    assert not result['level_clear_verified']


def test_runner_uncertain_click_is_not_retried(assets, tmp_path):
    driver = SyntheticDriver()
    driver.click = Mock(side_effect=RuntimeError('uncertain input outcome'))
    runner = Runner(max_steps=3, session_dir=tmp_path/'run', driver=driver,
                    config_path=assets[0], template_dir=assets[1])
    runner.acquire_mutex = lambda: None
    runner.wait = lambda seconds: None
    result = runner.run()
    driver.click.assert_called_once()
    assert result['pending'] and result['verified'] == 0
    assert result['failed']


def test_stale_frame_does_not_send_input():
    driver = object.__new__(WindowsDriver)
    driver.abort_pressed = lambda: False
    driver.validate = Mock()
    snapshot = Frame(frame(), (0, 0, 371, 709), time.monotonic()-3)
    with pytest.raises(RuntimeError, match='过期'):
        driver.click(snapshot, (100, 200))
    driver.validate.assert_not_called()


def test_cli_rejects_headless_without_explicit_input(assets):
    result = subprocess.run([sys.executable, 'desktop_app.py', '--headless', '--config', str(assets[0]),
                             '--templates', str(assets[1])], capture_output=True, text=True, encoding='utf-8')
    assert result.returncode != 0
    assert '--allow-input' in result.stderr


def test_self_check_requires_real_asset_presence(assets):
    result = subprocess.run([sys.executable, 'desktop_app.py', '--self-check', '--config', str(assets[0]),
                             '--templates', str(assets[1])], capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0
    assert json.loads(result.stdout)['ready_for_input'] is False


def test_cli_rejects_unconfirmed_calibration_even_with_allow_input(assets):
    result = subprocess.run([sys.executable, 'desktop_app.py', '--headless', '--allow-input',
                             '--config', str(assets[0]), '--templates', str(assets[1])],
                            capture_output=True, text=True, encoding='utf-8')
    assert result.returncode != 0
    assert 'ready_for_input is false' in result.stderr


@pytest.mark.parametrize('seconds', [float('nan'), float('inf'), 0, -1])
def test_runner_rejects_invalid_time_limits(assets, tmp_path, seconds):
    runner = Runner(max_seconds=seconds, session_dir=tmp_path/'run',
                    config_path=assets[0], template_dir=assets[1])
    with pytest.raises(ValueError):
        runner.run()
    assert not (tmp_path/'run').exists()


def test_unconfirmed_runner_never_constructs_native_driver(assets, tmp_path):
    runner = Runner(session_dir=tmp_path/'run', config_path=assets[0], template_dir=assets[1])
    with patch('standalone_runner.WindowsDriver', side_effect=AssertionError('No native driver')):
        result = runner.run()
    assert result['failed'] and result['clicks'] == 0 and not result['pending']


def test_image_inspection_is_offline_and_does_not_replace_outputs(assets, tmp_path):
    from inspect_frame import inspect
    input_path = tmp_path/'sample.png'
    cv2.imencode('.png', frame())[1].tofile(str(input_path))
    result = inspect(input_path, tmp_path/'inspection', *assets)
    assert result['visible_tiles'] == 3 and result['real_input_sent'] is False
    assert (tmp_path/'inspection/annotated.png').is_file()
    saved = (tmp_path/'inspection/state.json').read_bytes()
    with pytest.raises(FileExistsError):
        inspect(input_path, tmp_path/'inspection', *assets)
    assert (tmp_path/'inspection/state.json').read_bytes() == saved
