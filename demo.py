"""An offline three-action recognition/decision/verification demonstration."""
import argparse
import json
from pathlib import Path
import tempfile
from model import search, verify_transition
from synthetic import config, frame, write_fixture
from vision import observe, annotate


def run_demo(output=None):
    # No Windows driver is imported or called by this demo.
    result = {'mode': 'synthetic_offline', 'real_input_sent': False, 'verified_steps': 0, 'steps': []}
    with tempfile.TemporaryDirectory() as tmp:
        _, templates = write_fixture(Path(tmp) / 'assets')
        for clicked in range(3):
            image = frame(clicked)
            before = observe(image, config(), templates)
            plan = search(before, time_limit=1)
            if plan['status'] != 'proposal':
                raise RuntimeError('Synthetic scene did not produce the expected local route.')
            after = observe(frame(clicked+1), config(), templates)
            ok, reason = verify_transition(before, plan['next_action'], after)
            if not ok:
                raise RuntimeError(reason)
            result['verified_steps'] += 1
            result['steps'].append({'before_tray': before['tray']['items'],
                                    'action': plan['next_action'], 'after_tray': after['tray']['items']})
        result['final_tray'] = after['tray']['items']
        result['remaining_visible_tiles'] = len(after['tiles'])
        result['game_clear_claim'] = False
        if output:
            destination = Path(output)
            # Never replace an existing report or image directory.
            destination.mkdir(parents=True, exist_ok=False)
            annotate(frame(0), observe(frame(0), config(), templates), destination / 'synthetic-start.png')
            (destination / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new directory for artificial fixture output.')
    print(json.dumps(run_demo(parser.parse_args().output), ensure_ascii=False, indent=2))
