"""Frame-stability checks shared by the offline tests and Windows runner."""
import numpy as np


def stable(a, b, config):
    if a.shape != b.shape:
        return False, {'reason':'frame dimensions changed'}
    metrics=[]
    for x0,y0,x1,y1 in [config['board_roi'],config['tray_roi']]:
        delta=np.abs(a[y0:y1,x0:x1].astype(float)-b[y0:y1,x0:x1].astype(float))
        metrics.append({'mae':round(float(delta.mean()),4),
                        'changed_fraction':round(float((delta.max(axis=2)>22).mean()),5)})
    return all(m['mae']<2.0 and m['changed_fraction']<.025 for m in metrics), metrics
