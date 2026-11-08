"""Visibility-aware clothing and multi-view appearance banks over genuine crops."""
import numpy as np


def clothing_similarity(a,b):
    if a is None or b is None:return None
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=(32,) or b.shape!=(32,):return float(np.minimum(a,b).sum()/2)
    def soft(x):
        x=x.reshape(2,16);out=x.copy()
        # Adjacent hues/grey levels are close under lighting/exposure changes;
        # blue trousers and black trousers remain separate categories.
        out[:,:12]=.75*x[:,:12]+.125*np.roll(x[:,:12],1,axis=1)+.125*np.roll(x[:,:12],-1,axis=1)
        gray=x[:,12:];out[:,12:]=.6*gray+.2*np.c_[gray[:,:1],gray[:,:-1]]+.2*np.c_[gray[:,1:],gray[:,-1:]]
        out/=np.maximum(out.sum(axis=1,keepdims=True),1e-9)
        return out
    return float(np.minimum(soft(a),soft(b)).sum()/2)


def prototype_bank(samples,fallback):
    """Several supported poses, instead of averaging front/back into one vector."""
    if fallback is None:return np.empty((0,0),dtype='float32')
    bank=[fallback];good=[s for s in samples if s['quality']>=.3]
    if len(good)<2:return np.stack(bank)
    if len(good)>48:good=[good[i] for i in np.unique(np.linspace(0,len(good)-1,48).astype(int))]
    vectors=np.stack([s['feature'] for s in good]);remaining=set(range(len(good)))
    for _ in range(4):
        if not remaining:break
        seed=min(remaining,key=lambda k:max(float(vectors[k]@v) for v in bank))
        cluster=[k for k in sorted(remaining) if float(vectors[k]@vectors[seed])>=.8]
        remaining.difference_update(cluster)
        if len(cluster)<2 or max(good[k]['raw_pts'] for k in cluster)-min(good[k]['raw_pts'] for k in cluster)<.4:continue
        v=np.average(vectors[cluster],axis=0,weights=[good[k]['quality'] for k in cluster]);v/=max(np.linalg.norm(v),1e-9)
        bank.append(v)
    return np.stack(bank)


def gallery_similarity(a,b):
    aa,bb=a.get('appearance_bank'),b.get('appearance_bank')
    if aa is not None and bb is not None and aa.size and bb.size:return float(np.max(aa@bb.T))
    return float(a['vector']@b['vector']) if a['vector'] is not None and b['vector'] is not None else -1.
