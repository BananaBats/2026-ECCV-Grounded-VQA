"""Replay the actual saved-response SVAG recovery using caller-owned ADC.

Input is a historical-style, label-free experiment directory containing
inference_manifest.json and svag_v3/plans/<sample>/calls/*.json. The pure helper
pipeline.repair_plan does not require such journals. Default behavior is an
input check; --run requests only missing/ambiguous exact-frame grounding and,
when required, temporal/anchor repair, while locking previously selected IDs.
"""
import argparse,json,os
from pathlib import Path
from .common import read,queries,generation_client,save

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-experiment',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--sample',action='append',required=True);p.add_argument('--project',default=os.environ.get('GOOGLE_CLOUD_PROJECT'));p.add_argument('--location',default='global');p.add_argument('--run',action='store_true');a=p.parse_args(argv)
 source=a.source_experiment.resolve();manifest=read(source/'inference_manifest.json');rows=queries(source/'inference_manifest.json');selected=[r for r in rows if r['sample'] in a.sample]
 if len(selected)!=len(set(a.sample)):raise ValueError('Unknown selected query')
 if not a.run:print(json.dumps({'phase':'offline_check','selected':[r['sample'] for r in selected],'model_calls':0,'GT_inputs':[]}));return 0
 from ._legacy.svag_repair import historical_recovery as h
 h.PROJECT=a.project;client=generation_client(a.project,a.location);receipts=[]
 for row in selected:receipts.append(h.recover_sample(source,a.output_root,manifest,row,client))
 save(a.output_root/'recovery_index.json',{'samples':receipts});return 0
if __name__=='__main__':raise SystemExit(main())
