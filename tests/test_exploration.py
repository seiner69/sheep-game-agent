"""Synthetic risk-limit and probability-assisted choice tests."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from exploration import Explorer,covered_hints
from occlusion import OcclusionBeliefs,conservative_gain
from vision import image_read,observe
from model import verify_transition,search


def state(kinds,tray=()):
    return {'frame_id':'test','rules':{'tray_capacity':7,'match_size':3},
            'tray':{'items':list(tray),'complete':True},
            'tiles':[{'id':f'T{i}','kind':kind,'selectability':'visually_open',
                      'click_point':[100+i*40,200]} for i,kind in enumerate(kinds)]}


def node(identity,center,tile_id,gain):
    return {'id':identity,'center':center,'kind_hypothesis':None,
            'removal_effects':[{'remove_tile_id':tile_id,'marginal_exposure_gain':gain,'p_exposed_after':gain}]}


class ExplorationTests(unittest.TestCase):
    def test_exploration_uses_probabilities_without_clicking_a_dark_node(self):
        s=state(['a','b'])
        b={'hypotheses':[node('dark',[150,190],'T1',.8)]}
        plan=Explorer().choose(s,[],b)
        self.assertEqual(plan['next_action']['tile_id'],'T1')
        self.assertEqual(plan['next_action']['mode'],'explore')
        self.assertGreater(plan['next_action']['expected_exposure_utility'],0)

    def test_slot_reserve_is_hard_even_with_high_exposure_score(self):
        s=state(['x'],['a','b','c','d','e'])
        b={'hypotheses':[node('dark',[150,190],'T0',1)]}
        self.assertEqual(Explorer().choose(s,[],b)['status'],'pause')

    def test_known_clearance_is_allowed_at_five_slots(self):
        s=state(['a'],['a','a','b','c','d'])
        plan=Explorer().choose(s,[],{'hypotheses':[]})
        self.assertEqual(plan['next_action']['kind'],'a')
        self.assertEqual(plan['next_action']['mode'],'known_clearance')

    def test_probability_bonus_breaks_known_route_ties(self):
        s=state(['a','a','a'])
        plan=search(s,action_bonus={'T2':2},time_limit=1)
        self.assertEqual(plan['next_action']['tile_id'],'T2')

    def test_correlated_weak_labels_are_counted_once(self):
        s=state(['a']);after=state(['b']);after['tiles'][0]['click_point']=[150,200]
        nodes=[]
        for i,x in enumerate((150,169)):
            n=node(f'D{i}',[x,200],'T0',.5)
            n.update(p_real_dark_tile=.8,potential_blockers=[{'tile_id':'T0'}]);nodes.append(n)
        beliefs=OcclusionBeliefs();beliefs.update({'hypotheses':nodes},s,{'tile_id':'T0','kind':'a'},after)
        self.assertEqual((beliefs.alpha,beliefs.beta),(4,2))
        self.assertEqual(len(beliefs.evidence),1)

    def test_incomplete_tray_and_budget_block_exploration(self):
        s=state(['a']);s['tray']['complete']=False
        self.assertEqual(Explorer().choose(s,[])['status'],'pause')
        s['tray']['complete']=True
        e=Explorer(max_explorations=1);e.consecutive=1
        self.assertEqual(e.choose(s,[])['status'],'pause')

    def test_overlapping_hypotheses_do_not_multiply_reward(self):
        b={'hypotheses':[node('D1',[100,100],'T0',.4),node('D2',[119,100],'T0',.3),node('D3',[180,100],'T0',.2)]}
        gain,evidence=conservative_gain(b,'T0',Counter(),Counter())
        self.assertAlmostEqual(gain,.6)
        self.assertEqual(sum(e['used_in_utility'] for e in evidence),2)

    def test_blank_frame_does_not_invent_a_board(self):
        b=OcclusionBeliefs().infer(np.zeros((709,371,3),np.uint8),state(['a']),[],{})
        self.assertEqual(b['hypotheses'],[])


if __name__=='__main__':unittest.main()
