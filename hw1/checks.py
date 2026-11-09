"""Small independent numerical/protocol checks; run `python checks.py`."""
import numpy as np
import pandas as pd
import torch
from data_utils import TrainingData
from models import ItemKNN, IALS, EASE, FM, RatingRecencyALS
from evaluation import evaluate,recommend

def run_checks():
    f=pd.DataFrame({'u':[0,0,1,1,2,2], 'i':[0,1,0,2,1,3],
                    'rating':[5.,1.,4.,3.,5.,2.], 'timestamp':[1,2,3,4,5,6],
                    'format':['MP3']*6,'user':['a','a','b','b','c','c'],
                    'item':['w','x','w','y','x','z']})
    d=TrainingData(f,(3,4),10)
    assert d.features(np.array([0]),np.array([1])).max()<d.n_features
    knn=ItemKNN(neighbors=4,shrink=2).fit(d)
    x=d.positive.toarray();g=x.T@x;norm=np.sqrt((x*x).sum(0))
    expected=g/(norm[:,None]*norm[None,:]+2);np.fill_diagonal(expected,0)
    np.testing.assert_allclose(knn.score([0,1,2]),x@expected)
    ease=EASE(reg=3).fit(d)
    gram=x.T@x+3*np.eye(4)
    inv=np.linalg.inv(gram);b=-inv/np.diag(inv)[None,:];np.fill_diagonal(b,0)
    np.testing.assert_allclose(ease.weights,b,atol=1e-7)
    als=IALS(factors=2,reg=1,alpha=4,iterations=2).fit(d)
    for i in range(4):
        obs=f[f.i==i];y=als.user_factors[obs.u]
        a=als.user_factors.T@als.user_factors+4*y.T@y+np.eye(2)
        rhs=5*y.T@(obs.rating.to_numpy()>=4)
        np.testing.assert_allclose(a@als.item_factors[i],rhs,atol=1e-8)
    c=RatingRecencyALS(half_life=365)
    p=c.targets(d)
    assert p[0]>0 and p[1]<0 and p[3]==0
    # Independently sum pairwise FM interactions, then check one derivative.
    fm=FM(factors=3,epochs=1,batch_size=4).fit(d)
    ids=torch.as_tensor(d.features(np.array([0,1]),np.array([2,3])))
    linear=fm.linear(ids).sum((1,2));v=fm.embedding(ids)
    pair=torch.zeros(2)
    for j in range(ids.shape[1]):
        for k in range(j+1,ids.shape[1]):
            pair=pair+(v[:,j]*v[:,k]).sum(1)
    torch.testing.assert_close(fm.forward(ids),fm.bias+linear+pair)
    fm.forward(ids).sum().backward()
    index=int(ids[0,0]);analytic=fm.embedding.weight.grad[index,0].item()
    # Clear residual gradient from training before the derivative comparison.
    fm.embedding.weight.grad=None
    fm.forward(ids).sum().backward();analytic=fm.embedding.weight.grad[index,0].item()
    with torch.no_grad():
        original=fm.embedding.weight[index,0].item();eps=1e-3
        fm.embedding.weight[index,0]=original+eps;plus=fm.forward(ids).sum().item()
        fm.embedding.weight[index,0]=original-eps;minus=fm.forward(ids).sum().item()
        fm.embedding.weight[index,0]=original
    np.testing.assert_allclose(analytic,(plus-minus)/(2*eps),atol=2e-4)
    class Perfect:
        def score(self,users):return np.tile([0.,0.,5.,1.],(len(users),1))
    held=pd.DataFrame({'u':[0],'i':[2],'rating':[5.]})
    metrics,_=evaluate(Perfect(),held,d.seen,k=2)
    assert metrics['NDCG@10']==1 and metrics['Recall@10']==1
    assert set(recommend(Perfect(),[0],d.seen,k=2)[0])=={2,3}
    print('PASS: KNN, EASE, ALS normal equations, custom targets, FM pairs/gradient, ranking/masking')

if __name__=='__main__':
    torch.set_num_threads(2)
    run_checks()
