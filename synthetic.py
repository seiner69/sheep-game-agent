"""Deterministic artificial fixtures; no game artwork, screenshots or desktop I/O."""
from pathlib import Path
import json
import cv2
import numpy as np


def config():
    return {
        'window_title': 'Synthetic fixture - not a game window',
        'ready_for_input': False, 'frame_size': [371, 709],
        'board_roi': [52, 140, 319, 487], 'tray_roi': [22, 556, 349, 607],
        'minimum_score': 0.95, 'maximum_mae': 8, 'tray_scales': [1.0],
        'rules': {'tray_capacity': 7, 'match_size': 3},
        'hidden_regions': [], 'tray_validation_status': 'synthetic fixture only',
    }


def template(seed=17):
    return np.random.default_rng(seed).integers(35, 235, (28, 28, 3), dtype=np.uint8)


def frame(clicked=0):
    if clicked not in range(4):
        raise ValueError('Synthetic fixture has only three tiles.')
    image = np.full((709, 371, 3), (40, 100, 40), dtype=np.uint8)
    tile = template()
    for i in range(clicked, 3):
        x = 86 + i * 60
        image[186:214, x:x+28] = tile
    for i in range(clicked % 3):
        x = 36 + i * 48
        image[565:593, x:x+28] = tile
    return image


def write_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    templates = directory / 'templates'
    templates.mkdir()
    cv2.imencode('.png', template())[1].tofile(str(templates / 'synthetic_a.png'))
    cfg = directory / 'calibration.json'
    cfg.write_text(json.dumps(config(), ensure_ascii=False, indent=2), encoding='utf-8')
    return cfg, templates
