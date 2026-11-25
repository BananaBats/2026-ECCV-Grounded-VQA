"""Apply the actual GT-free structural SVAG repair to a saved model response.

Preserves target identities and predicted visibility/action intervals. Adds
initialization frames when necessary and records a soft-anchor-budget exception.
No spatial box is fabricated and no model call is made. Missing/ambiguous exact
anchor boxes must subsequently be grounded; contradictory timing is not guessed.
"""
import argparse,json
from pathlib import Path
from .common import read,save
from .svag_repair import repair_anchor_selection

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-selection',type=Path,required=True);p.add_argument('--allowed-frames',type=Path,required=True);p.add_argument('--num-frames',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
 frames=read(a.allowed_frames)
 if isinstance(frames,dict):frames=frames['allowed_anchor_frames']
 selection,changes=repair_anchor_selection(read(a.raw_selection),frames,a.num_frames);save(a.output,selection);save(a.output.with_suffix('.audit.json'),{'changes':changes,'model_calls':0,'GT_inputs':[],'spatial_grounding_required':True});print(json.dumps({'output':str(a.output),'anchors':len(selection['anchors']),'model_calls':0}));return 0
if __name__=='__main__':raise SystemExit(main())
