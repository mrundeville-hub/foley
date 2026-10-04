"""Render seed-42 comparisons against the four canonical sound references."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import foley as F
root=Path(__file__).resolve().parents[1]
out=root/'qa/sound_reference_comparison'
out.mkdir(parents=True,exist_ok=True)
results=[]
for item in json.loads((root/'references/audio_metrics.json').read_text()):
    frames=F.decode(item['source'])
    an=F.analyze(frames)
    plan=F.guess_plan(an)
    master,events,_=F.compose(an,plan,42)
    wav=out/(item['id']+'_after.wav')
    F.write_wav(str(wav),master)
    mono=master.mean(0)[::2]
    rate=F.SR/2
    windows=np.lib.stride_tricks.sliding_window_view(mono,2048)[::220]
    power=np.abs(np.fft.rfft(windows*np.hanning(2048)))**2
    freq=np.fft.rfftfreq(2048,1/rate)
    energy=power.sum(0)
    bands=[(20,150),(150,700),(700,3000),(3000,10000)]
    vals=[float(energy[(freq>=lo)&(freq<hi)].sum()) for lo,hi in bands]
    rms=float(np.sqrt(np.mean(master.astype(np.float64)**2)))
    peak=float(np.abs(master).max())
    result=dict(id=item['id'],vibe=plan['vibe'],events=len(events),rms_dbfs=round(20*np.log10(rms),2),crest_db=round(20*np.log10(peak/rms),2),centroid_median_hz=round(float(np.median((power*freq).sum(1)/(power.sum(1)+1e-12))),1),band_energy_pct={f'{lo}-{hi}Hz':round(v/sum(vals)*100,1) for (lo,hi),v in zip(bands,vals)})
    print(json.dumps(result),flush=True)
    results.append(result)
    if item['id']=='ref_03':
        F.mux(item['source'],str(wav),str(out/'ref_03_preview.mp4'))
(out/'after_metrics.json').write_text(json.dumps(results,indent=2))
