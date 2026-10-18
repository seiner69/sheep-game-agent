"""Scale regressions using generated noise patches, never game screenshots."""
import json
import time
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from configuration import validate_config
from exploration import Explorer
from occlusion import OcclusionBeliefs, face, grid_starts
from standalone_runner import Runner
from synthetic import config, template, write_fixture
from vision import detect, observe
from windows_driver import Frame


def scaled_frame(scale, dark=False):
    image = np.full((709, 371, 3), (40, 100, 40), np.uint8)
    tile = cv2.resize(template(), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if dark:
        tile = (tile * .55).astype(np.uint8)
    h, w = tile.shape[:2]
    for x in (86, 146, 206):
        image[186:186+h, x:x+w] = tile
    return image


@pytest.fixture
def assets(tmp_path):
    path, directory = write_fixture(tmp_path / 'assets')
    cfg = config()
    cfg['board_scales'] = [.8, .825, .85, .875, .9, 1.0]
    path.write_text(json.dumps(cfg), encoding='utf-8')
    return cfg, path, directory


@pytest.mark.parametrize('scale', [.8, .85, .9, 1.0])
def test_multiscale_recognition_and_cached_scale(assets, scale):
    cfg, _, directory = assets
    image = scaled_frame(scale)
    found = observe(image, cfg, directory)
    assert len(found['tiles']) == 3
    fitted = found['board_geometry']['template_scale']
    assert fitted == pytest.approx(scale, abs=.026)
    assert found['tray']['complete'] and found['tray']['items'] == []
    assert observe(image, cfg, directory, board_scale=fitted) == found
    assert Explorer().choose(found, [])['mode'] == 'known_clearance'


def test_previous_fixed_scale_really_misses_smaller_patches(assets):
    cfg, _, _ = assets
    hits, _ = detect(scaled_frame(.85), cfg['board_roi'], {'a': template()},
                     minimum_score=cfg['minimum_score'], maximum_mae=cfg['maximum_mae'])
    assert hits == []


def test_scaled_dark_patches_are_not_clickable(assets):
    cfg, _, directory = assets
    found = observe(scaled_frame(.85, dark=True), cfg, directory)
    assert found['tiles'] == []
    assert found['board_geometry']['template_scale'] is None
    assert Explorer().choose(found, [])['reason_code'] == 'board_unrecognized'


def test_slot_budget_has_separate_diagnostics(assets):
    cfg, _, directory = assets
    state = observe(scaled_frame(.85), cfg, directory)
    state['tiles'] = state['tiles'][:1]
    state['tray'] = {'items': ['a', 'b', 'c', 'd', 'e'], 'complete': True}
    plan = Explorer().choose(state, [])
    assert plan['reason_code'] == 'exploration_risk_budget'
    assert plan['diagnostics']['rejected_candidates'] == {'slot_reserve': 1}
    assert '5/7' in plan['reason']


def test_runner_refits_after_scale_change_without_native_input(assets, tmp_path):
    _, path, directory = assets
    runner = Runner(driver=Mock(), config_path=path, template_dir=directory,
                    session_dir=tmp_path/'session')
    runner.log = Mock()
    for scale in (.85, 1.0):
        state = runner.recognize(Frame(scaled_frame(scale), (0, 0, 371, 709), time.monotonic()))
        assert len(state['tiles']) == 3
        assert runner.board_scale == pytest.approx(scale, abs=.026)
    assert runner.log.call_count == 2
    runner.driver.click.assert_not_called()


def test_empty_board_runner_stops_without_clicking(assets, tmp_path):
    _, path, directory = assets
    blank = np.full((709, 371, 3), (40, 100, 40), np.uint8)
    driver = Mock()
    driver.abort_pressed.return_value = False
    driver.location.return_value = {'rect': [0, 0, 371, 709]}
    driver.capture.return_value = Frame(blank, (0, 0, 371, 709), time.monotonic())
    runner = Runner(driver=driver, config_path=path, template_dir=directory,
                    session_dir=tmp_path/'session')
    runner.acquire_mutex = lambda: None
    runner.wait = lambda seconds: None
    result = runner.run()
    driver.click.assert_not_called()
    assert result['clicks'] == 0 and not result['pending']
    assert not result['level_clear_verified']
    decision = json.loads((tmp_path/'session/001/decision.json').read_text(encoding='utf-8'))
    assert decision['reason_code'] == 'board_unrecognized'


def test_geometry_grid_uses_scale_and_visible_phase(assets):
    cfg, _, directory = assets
    image = scaled_frame(.85)
    state = observe(image, cfg, directory)
    scale = state['board_geometry']['template_scale']
    bounds = face([100, 200], scale)
    assert bounds[2]-bounds[0] == pytest.approx(38*scale)
    assert bounds[3]-bounds[1] == pytest.approx(41*scale)
    starts = grid_starts(state['tiles'], 0, 52, 300, scale)
    assert starts and all(52 <= x < 300 for x in starts)
    assert all(abs((b-a)-19*scale) <= 1 for a, b in zip(starts, starts[1:]))
    shifted = [{'click_point': [t['click_point'][0]+7, t['click_point'][1]]} for t in state['tiles']]
    assert grid_starts(shifted, 0, 59, 307, scale) == [x+7 for x in starts]
    belief = OcclusionBeliefs().infer(image, state, [], cfg)
    assert belief['board_scale'] == scale
    assert belief['hypotheses'] == []  # no dark evidence in artificial scene


@pytest.mark.parametrize('value', [[], [0], [True], [float('nan')], [float('inf')],
                                  [.49], [2.01], [1]*22, '0.85'])
def test_invalid_board_scales_rejected(value):
    cfg = config()
    cfg['board_scales'] = value
    with pytest.raises(ValueError, match='board_scales'):
        validate_config(cfg)
