"""Full-catalog deterministic top-K and macro-averaged ranking metrics."""
import numpy as np
import pandas as pd

def relevance(heldout, seen):
    relevant = heldout[heldout.rating >= 4]
    valid = np.asarray(seen[relevant.u.to_numpy(), relevant.i.to_numpy()]).ravel() == 0
    return {int(u):set(map(int,g.i)) for u,g in relevant[valid].groupby('u')}

def recommend(model, users, seen, k=10):
    scores = model.score(users)
    assert np.isfinite(scores).all()
    for row,u in enumerate(users):
        scores[row,seen.indices[seen.indptr[u]:seen.indptr[u+1]]] = -np.inf
    # Stable sorting breaks ties by ascending item ID, same for every model.
    top = np.argsort(-scores,axis=1,kind='stable')[:,:k]
    assert np.isfinite(np.take_along_axis(scores,top,axis=1)).all()
    return top

def evaluate(model, heldout, seen, k=10):
    truth = relevance(heldout,seen)
    users = sorted(truth)
    assert users
    rows=[]; all_items=set()
    discount=1/np.log2(np.arange(2,k+2))
    for start in range(0,len(users),32):
        batch=users[start:start+32]
        recs=recommend(model,batch,seen,k)
        for u,items in zip(batch,recs):
            hits=np.array([i in truth[u] for i in items])
            ideal=discount[:min(k,len(truth[u]))].sum()
            rows.append(dict(u=u,ndcg=float(hits@discount/ideal),recall=float(hits.sum()/len(truth[u]))))
            all_items.update(map(int,items))
    per_user=pd.DataFrame(rows)
    return {'NDCG@10':float(per_user.ndcg.mean()), 'Recall@10':float(per_user.recall.mean()),
            'Coverage@10':len(all_items)/seen.shape[1], 'users':len(users)},per_user

def paired_bootstrap(a,b,seed=42,n=2000):
    joined=a[['u','ndcg']].merge(b[['u','ndcg']],on='u',suffixes=('_a','_b'),validate='one_to_one')
    delta=(joined.ndcg_a-joined.ndcg_b).to_numpy()
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(n)])
    return dict(delta=float(delta.mean()),lower=float(np.quantile(means,.025)),upper=float(np.quantile(means,.975)),users=len(delta))
