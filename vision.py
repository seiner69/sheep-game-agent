"""A calibrated visual baseline, not a general detector or a probability model."""
from __future__ import annotations

import hashlib
from pathlib import Path
import cv2
import numpy as np

LABELS = {
    'shovel': '铲形工具', 'grass': '草', 'milk': '奶瓶', 'corn': '玉米',
    'wheat': '麦穗', 'yellow_item': '黄色图案', 'scissors': '剪刀',
    'cabbage': '白菜', 'stump': '树桩', 'cotton': '白色团状图案',
    'glove': '手套', 'carrot': '胡萝卜', 'fire': '火焰', 'bucket':'红桶', 'brush':'刷子',
    'yarn':'毛线球',
}


def image_read(path):
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f'Cannot read image: {path}')
    return result


def frame_id(image):
    return hashlib.sha256(image.tobytes()).hexdigest()[:20]


def detect(image, roi, templates, *, minimum_score=.85, maximum_mae=22, scales=(1.0,)):
    x0, y0, x1, y1 = roi
    crop = image[y0:y1, x0:x1]
    candidates = []
    variants = [(kind.split('__')[0], cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), scale)
                for kind,t in templates.items() for scale in scales]
    for kind, template, scale in variants:
        h, w = template.shape[:2]
        if h > crop.shape[0] or w > crop.shape[1]:
            continue
        scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(scores >= minimum_score)
        for x, y in sorted(zip(xs.tolist(), ys.tolist()), key=lambda p: -scores[p[1], p[0]]):
            cx, cy = x0+x+w/2, y0+y+h/2
            mae = float(np.abs(crop[y:y+h, x:x+w].astype(np.float32)-template.astype(np.float32)).mean())
            # NCC alone is invariant to brightness, so it can match dark covered
            # cards. Absolute color difference is a separate, necessary check.
            if mae > maximum_mae:
                continue
            candidates.append({'kind': kind, 'label_zh': LABELS.get(kind, kind),
                         'click_point': [round(cx, 1), round(cy, 1)],
                         'patch_bbox': [x0+x, y0+y, w, h],
                         'template_score': round(float(scores[y,x]), 4),
                         'color_mae': round(mae, 3),
                         'template_scale': scale,
                         'selectability': 'visually_open',
                         'confidence_semantics': 'Similarity scores, not calibrated probabilities.'})
    hits=[]
    for candidate in sorted(candidates,key=lambda t:-t['template_score']):
        if not any(t['kind']==candidate['kind']
                   and np.linalg.norm(np.subtract(t['click_point'],candidate['click_point']))<15 for t in hits):
            hits.append(candidate)
    # Conflicting labels at the same location cause abstention.
    ambiguous = set()
    for i, a in enumerate(hits):
        for j, b in enumerate(hits[:i]):
            if a['kind'] != b['kind'] and np.linalg.norm(np.subtract(a['click_point'], b['click_point'])) < 15:
                ambiguous.update((i, j))
    return [t for i, t in enumerate(hits) if i not in ambiguous], len(ambiguous)


def observe(image, config, template_dir):
    if [image.shape[1], image.shape[0]] != config['frame_size']:
        raise ValueError('Window size/DPI differs from calibration. Recalibrate; coordinates must not be reused.')
    templates = {p.stem: image_read(p) for p in sorted(Path(template_dir).glob('*.png'))}
    if not templates:
        raise ValueError('No templates are installed.')
    fid = frame_id(image)
    front, ambiguous = detect(image, config['board_roi'], templates,
                              minimum_score=config['minimum_score'], maximum_mae=config['maximum_mae'])
    front.sort(key=lambda t: (round(t['click_point'][1]/19), t['click_point'][0]))
    for n, tile in enumerate(front, 1):
        tile['id'] = f'T{n:02}'
        tile['id_scope'] = fid
    tray_hits, tray_ambiguous = detect(image, config['tray_roi'], templates,
                                      minimum_score=config['minimum_score'], maximum_mae=config['maximum_mae'],
                                      scales=config.get('tray_scales',[1.0]))
    tray_hits.sort(key=lambda t: t['click_point'][0])
    x0,y0,x1,y1 = config['tray_roi']
    tray_image = image[y0:y1,x0:x1]
    # Independent foreground check: unknown bright card faces may not be silently
    # interpreted as an empty tray. This heuristic still needs real tray samples.
    hsv = cv2.cvtColor(tray_image, cv2.COLOR_BGR2HSV)
    unexplained = cv2.inRange(hsv, np.array([0,0,145]), np.array([179,125,255]))
    for tile in tray_hits:
        x,y,w,h = tile['patch_bbox']
        a,b,c,d = max(0,x-x0-7),max(0,y-y0-7),min(x1-x0,x-x0+w+7),min(y1-y0,y-y0+h+7)
        unexplained[b:d,a:c] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(unexplained)
    unknown = [list(map(int, s[:4])) for s in stats[1:count]
               if s[cv2.CC_STAT_AREA] >= 80 and s[cv2.CC_STAT_WIDTH] >= 8 and s[cv2.CC_STAT_HEIGHT] >= 8]
    tray_complete = not unknown and tray_ambiguous == 0
    annotations = config.get('sample_annotations',{}) if fid == config.get('sample_frame_id') else {}
    edges = []
    covered = annotations.get('covered_observations', [])
    for c in covered:
        # Annotation names cover only plainly visible evidence, not a complete
        # inventory of occluded cards or complete prerequisites for a card.
        x,y,w,h = c['face_bbox']
        for tile in front:
            tx,ty,tw,th = tile['patch_bbox']
            if max(x,tx)<min(x+w,tx+tw) and max(y,ty)<min(y+h,ty+th):
                edges.append({'above': tile['id'], 'below': c['id'],
                              'evidence': 'overlap with manually annotated covered face in this sample'})
                tile['exposure_proxy'] = tile.get('exposure_proxy',0)+1
    return {
        'schema_version': 1, 'frame_id': fid,
        'source': {'window_title': config['window_title'], 'frame_size': config['frame_size'],
                   'coordinate_system': 'window screenshot pixels, origin at top-left',
                   'level': 2, 'calibration_scope': 'this window size and currently sampled art only'},
        'rules': config['rules'],
        'tray': {'items': [t['kind'] for t in tray_hits], 'complete': tray_complete,
                 'unknown_regions': unknown, 'roi_origin':[x0,y0],
                 'ambiguous_detections':tray_ambiguous,'independent_foreground_check': True,
                 'validation_status': config.get('tray_validation_status','empty tray sampled; occupied tray recognition not yet validated')},
        'tiles': front,
        'covered_observations': covered,
        'occlusion_edges': edges,
        'occlusion_graph_complete': False,
        'hidden_regions': config.get('hidden_regions',[]),
        'knowledge': {'whole_board_observed': False, 'remaining_total': None,
                      'remaining_kind_counts': None, 'absolute_layer_indices': None,
                      'covered_observations_exhaustive': False,
                      'unmatched_regions_may_contain_other_kinds': True,
                      'template_classes': list(templates),
                      'ambiguous_front_detections': ambiguous,
                      'detector_validated_on_independent_frames': False},
    }


def annotate(image, state, destination):
    canvas = image.copy()
    for x,y,w,h in state['tray'].get('unknown_regions',[]):
        # Tray unknown-region coordinates are relative to the calibrated ROI.
        tx,ty=state['tray'].get('roi_origin',[22,556])
        cv2.rectangle(canvas,(tx+x,ty+y),(tx+x+w,ty+y+h),(255,0,255),2)
        cv2.putText(canvas,'UNKNOWN',(tx+x,ty+y-3),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,0,255),1)
    for c in state['covered_observations']:
        x,y,w,h = c['face_bbox']
        cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,165,255),1)
        cv2.putText(canvas,c['id'],(x,y+10),cv2.FONT_HERSHEY_SIMPLEX,.27,(0,80,150),1)
    for tile in state['tiles']:
        x,y,w,h = tile['patch_bbox']
        cv2.rectangle(canvas,(x-2,y-2),(x+w+2,y+h+2),(70,50,245),1)
        cv2.putText(canvas,tile['id'],(x-2,y-4),cv2.FONT_HERSHEY_SIMPLEX,.31,(20,20,180),1,cv2.LINE_AA)
    cv2.imencode('.png', canvas)[1].tofile(str(destination))
