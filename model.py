"""Partial-observation state and bounded search. No hidden tile is invented."""
from __future__ import annotations

from collections import Counter
import time


def pick(tray: tuple[str, ...], kind: str, capacity: int = 7,
         match_size: int = 3) -> tuple[str, ...] | None:
    """Apply elimination first; None means the resulting tray is full/lost."""
    if len(tray) >= capacity:
        return None
    counts = Counter(tray)
    if any(n >= match_size for n in counts.values()):
        raise ValueError('Input must be a settled tray, with no unresolved match.')
    counts[kind] += 1
    if counts[kind] == match_size:
        del counts[kind]
    result = tuple(sorted(counts.elements()))
    return None if len(result) >= capacity else result


def search(state: dict, *, depth: int = 9, beam_width: int = 96,
           time_limit: float = 0.6, action_bonus: dict | None = None) -> dict:
    """Search only presently visible, visually open tiles.

    A plan is a local proposal, invalidated after the first real click.
    Covered/latent tiles never enter the search merely because an observed
    blocker disappears: the list of blockers may itself be incomplete.
    """
    if depth < 1 or beam_width < 1 or time_limit <= 0:
        raise ValueError('depth, beam_width, time_limit must be positive')
    base = {
        'method': 'bounded_beam_search_on_observed_tiles',
        'scope': 'local; no claim of full-level solvability',
        'frame_id': state['frame_id'],
        'reobserve_after_every_click': True,
        'hidden_tile_prior': None,
    }
    if not state['tray']['complete']:
        return base | {'status': 'pause', 'reason': 'Tray recognition is incomplete.'}
    capacity = state['rules']['tray_capacity']
    match_size = state['rules']['match_size']
    tray = tuple(sorted(state['tray']['items']))
    if len(tray) >= capacity or any(n >= match_size for n in Counter(tray).values()):
        return base | {'status': 'pause', 'reason': 'Tray is full or animation is unsettled.'}
    tiles = [t for t in state['tiles'] if t['selectability'] == 'visually_open'
             and t['kind'] is not None]
    if not tiles:
        return base | {'status': 'pause', 'reason': 'No recognized open tile; this is not proof of victory.'}

    # rank = cleared tiles, low occupancy, then a weak geometric reveal proxy.
    # The proxy does not infer the type or exact number of newly exposed tiles.
    def rank(mask, pocket, path):
        cleared = len(tray) + len(path) - len(pocket)
        exposure = sum(tiles[i].get('exposure_proxy', 0) for i in path)
        singles = sum(n == 1 for n in Counter(pocket).values())
        # Probabilistic exposure only breaks ties between comparable known routes.
        # Discount by order because only the first action is actually executed.
        belief_bonus=sum(min((action_bonus or {}).get(tiles[i]['id'],0),2)*(.8**n)
                         for n,i in enumerate(path))
        return cleared * 100 - len(pocket) * 15 - singles * 2 + min(exposure, 20) * .1 + min(belief_bonus,4)*.2

    initial = (0, tray, ())
    frontier = [initial]
    best = initial
    best_rank = rank(*initial)
    explored = 0
    reached_depth = 0
    deadline = time.perf_counter() + time_limit
    exhausted = False
    for step in range(depth):
        pool = {}
        for mask, pocket, path in frontier:
            for i, tile in enumerate(tiles):
                if mask & (1 << i):
                    continue
                new_pocket = pick(pocket, tile['kind'], capacity, match_size)
                explored += 1
                if new_pocket is not None:
                    child = (mask | (1 << i), new_pocket, path + (i,))
                    score = rank(*child)
                    key = (child[0], child[1])
                    if key not in pool:
                        pool[key] = (score, child)
                    if score > best_rank:
                        best, best_rank = child, score
                if time.perf_counter() >= deadline:
                    exhausted = True
                    break
            if exhausted:
                break
        if pool:
            frontier = [item[1] for item in sorted(pool.values(), key=lambda v: v[0], reverse=True)[:beam_width]]
            reached_depth = step + 1
        if exhausted or not pool:
            break
    stats = {'expanded_actions': explored, 'depth_reached': reached_depth,
             'time_budget_exhausted': exhausted, 'beam_width': beam_width}
    cleared = len(tray) + len(best[2]) - len(best[1])
    if not best[2] or cleared == 0:
        return base | {'status': 'pause', 'reason': 'No clearing route found within the search budget; not proof that none exists.',
                       'statistics': stats}
    sequence = []
    expected = tray
    for i in best[2]:
        tile = tiles[i]
        expected = pick(expected, tile['kind'], capacity, match_size)
        sequence.append({'tile_id': tile['id'], 'kind': tile['kind'],
                         'click_point': tile['click_point'],
                         'expected_tray_counts': dict(Counter(expected))})
    return base | {'status': 'proposal', 'statistics': stats,
                   'known_tiles_cleared_in_simulation': cleared,
                   'sequence_for_explanation_only': sequence,
                   'next_action': sequence[0],
                   'execution_gate': 'Needs a fresh matching frame, unobscured target window, and post-click tray verification.'}


def verify_transition(before: dict, action: dict, after: dict) -> tuple[bool, str]:
    """A mismatch must pause execution, never trigger a blind retry."""
    if not before['tray']['complete'] or not after['tray']['complete']:
        return False, 'Tray observation incomplete.'
    tile = next((t for t in before['tiles'] if t['id'] == action['tile_id']), None)
    if tile is None or tile['selectability'] != 'visually_open':
        return False, 'Action did not reference a recognized open tile.'
    if action.get('kind') != tile['kind'] or action.get('click_point') != tile['click_point']:
        return False, 'Action no longer agrees with the source tile.'
    expected = pick(tuple(before['tray']['items']), tile['kind'],
                    before['rules']['tray_capacity'], before['rules']['match_size'])
    if expected is None:
        return False, 'Action predicts a full tray.'
    if Counter(after['tray']['items']) != Counter(expected):
        return False, 'Actual tray differs from the predicted tray; stop, do not retry.'
    if before['frame_id'] == after['frame_id']:
        return False, 'No new visual observation.'
    return True, 'Observed tray matches this one-step prediction; replan from the new frame.'
