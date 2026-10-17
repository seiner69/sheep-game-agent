"""Inspect an explicitly supplied image offline; no window capture or input."""
import argparse
import json
from pathlib import Path
from configuration import load_assets
from model import search
from vision import annotate, image_read, observe


def inspect(image_path, output, config_path=None, template_dir=None):
    config, templates, _ = load_assets(config_path, template_dir)
    state = observe(image_read(image_path), config, templates)
    decision = search(state, time_limit=1)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination/'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
    annotate(image_read(image_path), state, destination/'annotated.png')
    return {'real_input_sent': False, 'visible_tiles': len(state['tiles']),
            'tray_complete': state['tray']['complete'], 'decision_status': decision['status']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New private output directory, e.g. local/inspection-1')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--templates', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.image, args.output, args.config, args.templates), ensure_ascii=False))


if __name__ == '__main__':
    main()
