"""Risk-bounded information gathering. Hints are hypotheses, never legal moves."""
from collections import Counter
import math
import cv2
import numpy as np
from model import pick,search


def covered_hints(image,config,templates,front,board_scale=1.0):
    x0,y0,x1,y1=config['board_roi'];roi=image[y0:y1,x0:x1]
    candidates=[]
    for kind,t in templates.items():
        kind=kind.split('__')[0]
        if board_scale != 1.0:
            t=cv2.resize(t,None,fx=board_scale,fy=board_scale,interpolation=cv2.INTER_CUBIC)
        h,w=t.shape[:2]
        # Partial masks can find a recognisable dark fragment beside a top card.
        crops=[(0,0,w,h),(0,0,math.ceil(w*.45),h),(w-math.ceil(w*.45),0,w,h),
               (0,0,w,math.ceil(h*.45)),(0,h-math.ceil(h*.45),w,h)]
        for ax,ay,bx,by in crops:
            patch=t[ay:by,ax:bx]
            if patch.std()<22:continue
            scores=cv2.matchTemplate(roi,patch,cv2.TM_CCOEFF_NORMED)
            for _ in range(5):
                _,score,_,(px,py)=cv2.minMaxLoc(scores)
                if score<.85:break
                ph,pw=patch.shape[:2]
                observed=roi[py:py+ph,px:px+pw].astype(float)
                reference=patch.astype(float)
                centered=reference-reference.mean()
                scale=float(((observed-observed.mean())*centered).sum()/(centered**2).sum())
                offset=float(observed.mean()-scale*reference.mean())
                residual=float(np.abs(observed-(reference*scale+offset)).mean())
                center=[x0+px-ax+w/2,y0+py-ay+h/2]
                scores[max(0,py-12):py+13,max(0,px-12):px+13]=0
                if not (.35<scale<.90 and abs(offset)<35 and residual<15):continue
                if any(np.linalg.norm(np.subtract(center,f['click_point']))<17*board_scale for f in front):continue
                if any(np.linalg.norm(np.subtract(center,c['center']))<15*board_scale for c in candidates):continue
                candidates.append({'kind_hypothesis':kind,'center':center,'score':round(score,3),
                                   'partial':(ax,ay,bx,by)!=(0,0,w,h),'brightness_scale':round(scale,3),
                                   'semantics':'visual hint only; not clickable and not counted as known inventory'})
    return candidates


def overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


class Explorer:
    def __init__(self,reserve=2,max_explorations=6):
        self.reserve=reserve;self.max_explorations=max_explorations
        self.consecutive=0;self.history=[];self.revealed=Counter()

    def record(self,before,action,after):
        clears=len(before['tray']['items'])+1-len(after['tray']['items'])
        if clears:self.consecutive=0
        elif action.get('mode')=='explore':self.consecutive+=1
        new=[]
        for tile in after['tiles']:
            if not any(t['kind']==tile['kind'] and np.linalg.norm(np.subtract(t['click_point'],tile['click_point']))<8
                       for t in before['tiles']):new.append(tile['kind'])
        self.revealed.update(new)
        self.history.append({'clicked':action['kind'],'mode':action.get('mode'),'newly_visible':new,'cleared':clears})

    def choose(self,state,hints,belief=None):
        from occlusion import conservative_gain
        pocket=Counter(state['tray']['items']);visible=Counter(t['kind'] for t in state['tiles'])
        bonuses={t['id']:conservative_gain(belief,t['id'],pocket,visible)[0] for t in state['tiles']}
        plan=search(state,time_limit=.35,action_bonus=bonuses)
        if plan['status']=='proposal':
            plan['mode']='known_clearance';plan['next_action']['mode']='known_clearance'
            gain,evidence=conservative_gain(belief,plan['next_action']['tile_id'],pocket,visible)
            plan['next_action'].update(expected_exposure_utility=round(gain,4),probability_evidence=evidence)
            return plan
        if not state['tray']['complete']:
            return plan
        if not any(t['selectability']=='visually_open' and t['kind'] for t in state['tiles']):
            return {'status':'pause','reason_code':'board_unrecognized',
                    'reason':'未识别到可点击明牌：请检查是否在第二关牌面、图案尺寸或模板是否适配；不能据此判断通关。'}
        if self.consecutive>=self.max_explorations:
            return {'status':'pause','reason_code':'exploration_limit',
                    'reason':f'连续探索 {self.consecutive} 步仍未消除，达到本次探索上限。'}
        tray=tuple(state['tray']['items']);pocket=Counter(tray)
        capacity=state['rules']['tray_capacity'];visible=Counter(t['kind'] for t in state['tiles'])
        candidates=[]
        fronts=[t for t in state['tiles'] if t['selectability']=='visually_open' and t['kind']]
        scale=state.get('board_geometry',{}).get('template_scale') or 1.0
        fw,fh=38*scale,42*scale
        def face_at(center):
            x,y=center;return (x-fw/2,y-fh/2,x+fw/2,y+fh/2)
        rejected=Counter()
        for tile in fronts:
            expected=pick(tray,tile['kind'],capacity,state['rules']['match_size'])
            if expected is None:
                rejected['would_fill_tray']+=1;continue
            if len(expected)>capacity-self.reserve:
                rejected['slot_reserve']+=1;continue
            x,y=tile['click_point'];face=face_at(tile['click_point'])
            information=0.;useful=[]
            for hint in hints:
                hidden=face_at(hint['center'])
                area=overlap(face,hidden)
                if not (45*scale**2<area<1300*scale**2):continue
                blockers=sum(overlap(face_at(f['click_point']),hidden)>45*scale**2 for f in fronts)
                kind=hint['kind_hypothesis']
                support=pocket[kind]+visible[kind]
                usefulness=7 if support>=2 else 3 if support==1 else 1
                information+=usefulness/max(blockers,1)
                useful.append({'hint':kind,'observed_blockers':blockers,'support':support})
            stack=(x<90 or x>280) and y>380
            # Central removal generally reveals neighbors; stacks reveal one next
            # tile. Both are scored as information, not invented tile identities.
            generic=1.0 if stack else 1.5
            probabilistic_gain,probability_evidence=conservative_gain(belief,tile['id'],pocket,visible)
            same_kind_support=visible[tile['kind']]-1+pocket[tile['kind']]
            new_kind=tile['kind'] not in pocket
            score=information*2+min(probabilistic_gain,2)*12+generic+min(same_kind_support,2)*3+min(pocket[tile['kind']],2)*4
            score-=len(expected)*2+new_kind*3
            # Near the slot limit demand meaningful evidence of revealing a
            # useful kind; an uninformative blind draw is not forced.
            if len(expected)>=5 and information<2 and probabilistic_gain<.2:
                rejected['insufficient_reveal_evidence']+=1;continue
            candidates.append({'tile_id':tile['id'],'kind':tile['kind'],'click_point':tile['click_point'],
                               'expected_tray_counts':dict(Counter(expected)),'mode':'explore',
                               'score':round(score,3),'information_hints':useful,
                               'probability_evidence':probability_evidence,'expected_exposure_utility':round(probabilistic_gain,4),
                               'risk':'unknown hidden face; heuristic score is not a win probability',
                               'reserved_slots_after_action':capacity-len(expected)})
        if not candidates:
            reason=(f'已识别明牌 {len(fronts)} 张、槽位 {len(tray)}/{capacity}；搜索未找到消除路线。')
            if rejected['slot_reserve']:
                reason+=f'探索需保留 {self.reserve} 格，{rejected["slot_reserve"]} 个候选超出限制。'
            if rejected['insufficient_reveal_evidence']:
                reason+=f'{rejected["insufficient_reveal_evidence"]} 个候选缺少足够揭露线索。'
            if rejected['would_fill_tray']:
                reason+=f'{rejected["would_fill_tray"]} 个候选会填满槽位。'
            return {'status':'pause','reason_code':'exploration_risk_budget','reason':reason,
                    'diagnostics':{'open_tiles':len(fronts),'tray_used':len(tray),'capacity':capacity,
                                   'reserve':self.reserve,'rejected_candidates':dict(rejected),
                                   'known_search':plan.get('statistics',{})}}
        candidates.sort(key=lambda a:-a['score'])
        return {'status':'proposal','mode':'explore','frame_id':state['frame_id'],
                'next_action':candidates[0],'alternatives':candidates[:5],
                'consecutive_explorations':self.consecutive,'hidden_information_known':False,
                'reobserve_after_every_click':True}
