"""Exact positive-extent and output-support controls used in validation.

Function bodies retained from the experiment implementation.
"""
from bisect import bisect_left
from .core import sane_yxyx

def counterfactual_correction(sam, sparse, references):
    """Remove positive halfextent changes, keeping center motion and shrinking."""
    residual = {}
    for frame in references:
        r = [sparse[frame][i]-sam[frame][i] for i in range(4)]
        modified = [0.0]*4
        for lo, hi in [(0, 2), (1, 3)]:
            center = (r[lo]+r[hi])/2
            halfextent = min(0.0, (r[hi]-r[lo])/2)
            modified[lo], modified[hi] = center-halfextent, center+halfextent
        residual[frame] = modified
    corrected = {}
    for frame, box in sam.items():
        index = bisect_left(references, frame)
        previous = references[index-1] if index else None
        following = references[index] if index < len(references) else None
        if frame in residual:
            # At a reference this box is contained in G: same G center and
            # min(SAM,G) extents, hence G's input validity is preserved.
            corrected[frame] = [box[i]+residual[frame][i] for i in range(4)]
            continue
        if previous is not None and following is not None:
            w = (frame-previous)/(following-previous)
            r = [residual[previous][i]*(1-w)+residual[following][i]*w for i in range(4)]
        elif previous is not None:
            r = residual[previous]
        elif following is not None:
            r = residual[following]
        else:
            corrected[frame] = list(box)
            continue
        try:
            corrected[frame] = sane_yxyx([box[i]+r[i] for i in range(4)], min_size=2.0)
        except ValueError:
            corrected[frame] = list(box)
    return corrected

def fixed_support(merged, canonical):
    """Use the canonical output mask and its exact observed gap endpoints."""
    frames = sorted(merged)
    result = {}
    for frame in canonical:
        if frame in merged:
            result[frame] = list(merged[frame])
            continue
        index = bisect_left(frames, frame)
        if index == 0 or index == len(frames):
            raise ValueError('Canonical fusion contains an extrapolated output frame')
        a, b = frames[index-1], frames[index]
        w = (frame-a)/(b-a)
        result[frame] = [merged[a][i]*(1-w)+merged[b][i]*w for i in range(4)]
    if set(result) != set(canonical):
        raise ValueError('Fixed-support counterfactual changed output frames')
    return result
