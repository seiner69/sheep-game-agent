"""Validate user-owned local calibration/assets before any desktop operation."""
import json
import math
from pathlib import Path
import sys
from vision import image_read

RESOURCE = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))


def validate_config(config):
    size = config.get('frame_size')
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or not 128 <= v <= 8192 for v in size):
        raise ValueError('frame_size must contain two integers in [128, 8192].')
    if not isinstance(config.get('window_title'), str) or not config['window_title'].strip():
        raise ValueError('window_title must be a nonempty exact window title.')
    if type(config.get('ready_for_input', False)) is not bool:
        raise ValueError('ready_for_input must be true or false.')
    for name in ('board_roi', 'tray_roi'):
        roi = config.get(name)
        if not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi):
            raise ValueError(f'{name} must be [left, top, right, bottom] integers.')
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
            raise ValueError(f'{name} must be nonempty and inside frame_size.')
    a, b = config['board_roi'], config['tray_roi']
    if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
        raise ValueError('board_roi and tray_roi must not overlap.')
    for name, low, high in (('minimum_score', 0.5, 1), ('maximum_mae', 0, 255)):
        v = config.get(name)
        if type(v) not in (float, int) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f'{name} is outside its allowed range.')
    rules = config.get('rules', {})
    # Search, exploration and GUI currently implement the same seven-slot rule.
    if rules.get('tray_capacity') != 7 or rules.get('match_size') != 3:
        raise ValueError('This version supports tray_capacity=7 and match_size=3 only.')
    scales = config.get('tray_scales', [1.0])
    if not isinstance(scales, list) or not 1 <= len(scales) <= 12 or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0.2 <= v <= 4 for v in scales):
        raise ValueError('tray_scales must contain 1-12 finite numbers in [0.2, 4].')
    return config


def load_assets(config_path=None, template_dir=None):
    config_path = Path(config_path) if config_path else RESOURCE / 'local/calibration.json'
    template_dir = Path(template_dir) if template_dir else RESOURCE / 'local/templates'
    if not config_path.is_file():
        raise ValueError('Missing local calibration. Follow README; synthetic demo: python demo.py')
    config = validate_config(json.loads(config_path.read_text(encoding='utf-8')))
    paths = sorted(template_dir.glob('*.png'))
    if not 1 <= len(paths) <= 128:
        raise ValueError('Provide 1-128 locally owned PNG templates; none are bundled.')
    templates = {}
    for path in paths:
        if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('Templates must be regular files no larger than 2 MiB.')
        image = image_read(path)
        h, w = image.shape[:2]
        x0, y0, x1, y1 = config['board_roi']
        if min(h, w) < 4 or h > y1-y0 or w > x1-x0 or image.std() < 5:
            raise ValueError('Template is blank, too small, or larger than the board region.')
        templates[path.stem] = image
    return config, template_dir, templates
