"""Explicit, uncalibrated probability beliefs about which cards cover dark faces."""
import itertools
import math
import cv2
import numpy as np
from exploration import overlap


def face(center):
    x,y=center;return (x-19,y-20,x+19,y+21)


def conservative_gain(belief,tile_id,pocket,visible):
    """Do not count nearby segmentation alternatives as independent rewards."""
    values=[]
    for node in (belief or {}).get('hypotheses',[]):
        effect=next((e for e in node['removal_effects'] if e['remove_tile_id']==tile_id),None)
        if effect is None:continue
        kind=node['kind_hypothesis'];support=pocket[kind]+visible[kind] if kind else 0
        utility=4 if support>=2 else 2 if support==1 else 1
        values.append((effect['marginal_exposure_gain']*utility,node,effect))
    kept=[];evidence=[]
    for value,node,effect in sorted(values,key=lambda v:-v[0]):
        # Nearby hypotheses may refer to the same face. Keeping only the best
        # local reward is deliberately conservative; this is not a tile count.
        duplicate=any(np.linalg.norm(np.subtract(node['center'],n['center']))<27 for n in kept)
        evidence.append({'dark_node':node['id'],**effect,'used_in_utility':not duplicate})
        if not duplicate:kept.append(node)
    return sum(value for value,node,effect in values if node in kept),evidence


class OcclusionBeliefs:
    def __init__(self):
        # Weak prior for successful exposure after the sole observed blocker is
        # removed. This includes unseen blockers, segmentation and detector error.
        self.alpha=3.;self.beta=2.;self.evidence=[]

    def infer(self,image,state,hints,config):
        h,w=image.shape[:2]
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny(gray,45,100)>0
        dark=np.linalg.norm(image.astype(float)-np.array([123,153,147]),axis=2)<23
        known=np.zeros((h,w),np.uint8)
        fronts=state['tiles']
        for tile in fronts:
            x0,y0,x1,y1=map(round,face(tile['click_point']))
            known[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=1
        hypotheses=[]
        # Layout anchors are calibrated from this game's regular 19px half-grid.
        # They are hypotheses; a candidate must have actual visible dark evidence.
        for y0 in range(141,450,19):
            for x0 in range(54,300,19):
                x1,y1=x0+38,y0+41;center=[x0+19,y0+20]
                if x1>w or y1>h:continue
                if any(np.linalg.norm(np.subtract(center,t['click_point']))<14 for t in fronts):continue
                visible=known[y0+2:y1-2,x0+2:x1-2]==0
                vf=float(visible.mean())
                if not .15<vf<.92:continue
                dark_pixels=dark[y0+2:y1-2,x0+2:x1-2]&visible
                dark_fraction=float(dark_pixels.sum()/max(visible.sum(),1))
                if dark_pixels.sum()<45 or dark_fraction<.22:continue
                border=[]
                for xa,ya,xb,yb in [(x0-1,y0,x0+2,y1),(x1-2,y0,x1+1,y1),
                                    (x0,y0-1,x1,y0+2),(x0,y1-2,x1,y1+1)]:
                    v=known[ya:yb,xa:xb]==0
                    if v.sum()>=8:border.append(float((edges[ya:yb,xa:xb]&v).sum()/v.sum()))
                if len(border)<2 or sorted(border,reverse=True)[1]<.15:continue
                edge_support=min(1,float(np.mean(border))*2.5)
                p_real=min(.96,max(.12,.10+.55*dark_fraction+.35*edge_support))
                blockers=[]
                for tile in fronts:
                    rect=face(tile['click_point'])
                    if overlap(rect,(x0,y0,x1,y1))<16:continue
                    # Marginalize small localization uncertainty using a nine-point
                    # geometry prior. Scores remain uncalibrated estimates.
                    intersections=[overlap(rect,(x0+dx,y0+dy,x1+dx,y1+dy))>20
                                   for dx,dy in itertools.product((-2,0,2),repeat=2)]
                    p=.05+.90*sum(intersections)/len(intersections)
                    blockers.append({'tile_id':tile['id'],'kind':tile['kind'],
                                     'p_above_given_dark_tile':round(p,4),
                                     'overlap_area':round(overlap(rect,(x0,y0,x1,y1)),1),
                                     'evidence':'bright face overlapping a dark-face geometry hypothesis'})
                if not blockers:continue
                blockers.sort(key=lambda b:-b['p_above_given_dark_tile'])
                blockers=blockers[:5]
                reliability=self.alpha/(self.alpha+self.beta)
                p_no_other=reliability*(.4+.6*vf)
                base=p_real*p_no_other*math.prod(1-b['p_above_given_dark_tile'] for b in blockers)
                effects=[]
                for b in blockers:
                    after=p_real*p_no_other*math.prod(1-other['p_above_given_dark_tile']
                                                    for other in blockers if other is not b)
                    effects.append({'remove_tile_id':b['tile_id'],'p_exposed_after':round(after,4),
                                    'marginal_exposure_gain':round(after-base,4)})
                nearby=[hint for hint in hints if np.linalg.norm(np.subtract(hint['center'],center))<15]
                kind=max(nearby,key=lambda a:a['score'])['kind_hypothesis'] if nearby else None
                scenarios=[]
                for bits in itertools.product((False,True),repeat=len(blockers)):
                    probability=math.prod(b['p_above_given_dark_tile'] if bit else 1-b['p_above_given_dark_tile']
                                          for b,bit in zip(blockers,bits))
                    scenarios.append({'above_tile_ids':[b['tile_id'] for b,bit in zip(blockers,bits) if bit],
                                      'p_conditional_on_dark_tile':round(probability,5)})
                hypotheses.append({'id':f'D{x0}_{y0}','center':center,'face_bbox':[x0,y0,38,41],
                                   'p_real_dark_tile':round(p_real,4),'kind_hypothesis':kind,
                                   'potential_blockers':blockers,'p_no_unobserved_blocker':round(p_no_other,4),
                                   'p_already_exposed':round(base,4),'removal_effects':effects,
                                   'blocker_set_scenarios':sorted(scenarios,key=lambda a:-a['p_conditional_on_dark_tile'])[:6],
                                   'evidence':{'visible_fraction':round(vf,3),'dark_fraction':round(dark_fraction,3),
                                               'edge_support':round(edge_support,3)}})
        hypotheses.sort(key=lambda n:-n['p_real_dark_tile'])
        return {'calibration_status':'heuristic geometric prior; NOT calibrated statistical probabilities',
                'frame_id':state['frame_id'],
                'dependency_assumption':'blocker edges conditionally independent for the displayed scenario approximation',
                'inventory_warning':'overlapping hypotheses may describe the same physical tile; do not sum them as tile count',
                'reveal_reliability_beta':{'alpha':self.alpha,'beta':self.beta,'mean':self.alpha/(self.alpha+self.beta)},
                'hypotheses':hypotheses[:32]}

    def update(self,belief,before,action,after):
        new=[t for t in after['tiles'] if not any(old['kind']==t['kind'] and
             np.linalg.norm(np.subtract(old['click_point'],t['click_point']))<8 for old in before['tiles'])]
        used=[]
        for node in sorted(belief['hypotheses'],key=lambda n:-n['p_real_dark_tile']):
            blockers=node['potential_blockers']
            if node['p_real_dark_tile']<.55 or len(blockers)!=1 or blockers[0]['tile_id']!=action['tile_id']:continue
            if any(np.linalg.norm(np.subtract(node['center'],center))<27 for center in used):continue
            used.append(node['center'])
            exposed=any(np.linalg.norm(np.subtract(t['click_point'],node['center']))<13 for t in new)
            if exposed:self.alpha+=1
            else:self.beta+=1
            self.evidence.append({'node':node['id'],'removed':action['kind'],'exposure_observed':exposed,
                                  'label_quality':'automatic visual outcome; non-detection is not proof of absence'})
