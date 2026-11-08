"""Read-only private review of existing FULL tracks and genuine OSNet features."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from shadowtrack.mvp.config import Settings
from shadowtrack.mvp.db import connect, rows
from shadowtrack.detection.common import atomic_json


def main():
    s=Settings.environment();out=s.artifacts/'mtmc';out.mkdir(exist_ok=True)
    with connect(s.database,readonly=True) as db:
        tracks=rows(db,'SELECT t.*,e.vector,e.evidence,e.samples FROM local_track t JOIN embedding e ON e.track_id=t.id ORDER BY t.id')
    matrix=np.stack([np.frombuffer(t.pop('vector'),'<f4') for t in tracks]);scores=matrix@matrix.T
    pairs=[]
    for i,j in combinations(range(len(tracks)),2):
        a,b=tracks[i],tracks[j]
        if a['camera_id']==b['camera_id']:continue
        gap=max(a['start_time'],b['start_time'])-min(a['end_time'],b['end_time'])
        if gap>35 or gap < -60:continue
        pairs.append({'a':a['id'],'b':b['id'],'camera_pair':sorted([a['camera_id'],b['camera_id']]),'similarity':float(scores[i,j]),'gap':gap})
    pairs.sort(key=lambda p:-p['similarity']);lookup={t['id']:t for t in tracks}
    chosen=[]
    for cam_a,cam_b in combinations('12345',2):
        ranked=[p for p in pairs if p['camera_pair']==[cam_a,cam_b]]
        chosen.extend(ranked[:4])
        print(cam_a,cam_b,[(lookup[p['a']]['display_id'],lookup[p['b']]['display_id'],round(p['similarity'],3),round(p['gap'],1)) for p in ranked[:5]])
    atomic_json(out/'appearance_research.json',{'source':'existing real multi-crop OSNet; no manual identity labels','candidates':pairs,'selected':chosen})
    for page in range((len(chosen)+7)//8):
        sheet=Image.new('RGB',(1100,8*205),'#141914');draw=ImageDraw.Draw(sheet)
        for row,p in enumerate(chosen[page*8:page*8+8]):
            draw.text((8,row*205+4),f"{p['camera_pair']} cosine={p['similarity']:.3f} gap={p['gap']:.2f}s UNCHECKED",fill='white')
            for col,tid in enumerate([p['a'],p['b']]):
                t=lookup[tid];ev=json.loads(t['evidence']);draw.text((col*550+8,row*205+24),f"{t['display_id']} {t['start_time']:.2f}..{t['end_time']:.2f} {tid}",fill='#d4ff38')
                for k,e in enumerate(ev['samples'][:4]):
                    with Image.open(s.artifacts/'reid/crops'/e['path']) as im:
                        im.thumbnail((127,155));sheet.paste(im,(col*550+k*134+8,row*205+44))
        sheet.save(out/f'candidate_sheet_{page}.jpg',quality=90)


if __name__=='__main__':main()
