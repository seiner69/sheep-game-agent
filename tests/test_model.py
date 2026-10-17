from collections import Counter
from copy import deepcopy
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import pick, search, verify_transition


def state(kinds, tray=()):
    return {'frame_id':'one', 'rules':{'tray_capacity':7,'match_size':3},
            'tray':{'items':list(tray),'complete':True},
            'tiles':[{'id':str(i),'kind':k,'click_point':[i,0],
                      'selectability':'visually_open'} for i,k in enumerate(kinds)]}


class StateTests(unittest.TestCase):
    def test_third_tile_clears(self):
        self.assertEqual(pick(('a','b','a'),'a'),('b',))

    def test_seventh_tile_can_clear_before_overflow(self):
        self.assertEqual(pick(('a','a','b','b','c','d'),'a'),('b','b','c','d'))

    def test_seventh_nonmatch_loses(self):
        self.assertIsNone(pick(('a','a','b','b','c','d'),'e'))

    def test_lost_state_cannot_recover(self):
        self.assertIsNone(pick(('a','a','b','b','c','d','e'),'a'))

    def test_unknown_or_covered_not_searchable(self):
        s=state(['a','a','a',None])
        s['tiles'][2]['selectability']='covered'
        self.assertEqual(search(s)['status'],'pause')

    def test_incomplete_tray_stops(self):
        s=state(['a','a','a']); s['tray']['complete']=False
        self.assertEqual(search(s)['status'],'pause')

    def test_bounded_route_never_overflows(self):
        s=state(['x','a','b','a','a','b','b'],['z','z'])
        result=search(s,time_limit=2)
        self.assertEqual(result['status'],'proposal')
        pocket=tuple(s['tray']['items']); seen=set()
        for action in result['sequence_for_explanation_only']:
            self.assertNotIn(action['tile_id'],seen); seen.add(action['tile_id'])
            pocket=pick(pocket,action['kind'])
            self.assertIsNotNone(pocket)
            self.assertEqual(dict(Counter(pocket)),action['expected_tray_counts'])

    def test_failed_click_is_not_success(self):
        before=state(['a','a','a']); plan=search(before)
        after=deepcopy(before); after['frame_id']='two'
        self.assertFalse(verify_transition(before,plan['next_action'],after)[0])

    def test_verifies_expected_bag_not_slot_order(self):
        before=state(['a'],['a','a','b']); after=deepcopy(before)
        after['frame_id']='two'; after['tray']['items']=['b']
        action={'tile_id':'0','kind':'a','click_point':[0,0]}
        self.assertTrue(verify_transition(before,action,after)[0])

    def test_no_front_tiles_is_not_victory(self):
        self.assertEqual(search(state([]))['status'],'pause')


if __name__=='__main__': unittest.main()
