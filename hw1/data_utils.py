"""Download, deterministic cleaning, global chronological split, train-only features."""
from pathlib import Path
import gzip, hashlib, json, os, urllib.request
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE = 'https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/'
BASE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/'
URLS = {'Digital_Music_5.json.gz': BASE+'categoryFilesSmall/Digital_Music_5.json.gz',
        'meta_Digital_Music.json.gz': BASE+'metaFiles2/meta_Digital_Music.json.gz'}
SEED = 42

def download(directory=None):
    directory = Path(directory or os.environ.get('RECSYS_DATA', '.cache/amazon'))
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(__file__).with_name('data_checksums.json')
    expected = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    hashes = {}
    for name, url in URLS.items():
        path = directory / name
        if not path.exists():
            tmp = path.with_suffix('.partial')
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in expected and expected[name] != hashes[name]:
            raise ValueError(f'Checksum mismatch: {name}')
    return directory, hashes

def load(directory):
    records = []
    with gzip.open(Path(directory)/'Digital_Music_5.json.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            style = r.get('style') or {}
            records.append(dict(user=r.get('reviewerID'), item=r.get('asin'),
                                rating=r.get('overall'), timestamp=r.get('unixReviewTime'),
                                format=str(style.get('Format:', 'unknown')).strip()))
    raw = pd.DataFrame(records)
    valid = raw.dropna(subset=['user','item','rating','timestamp'])
    valid = valid[valid.rating.between(1,5) & (valid.timestamp > 0)]
    # First rating is the only feedback available when an item first becomes known.
    clean = valid.sort_values(['timestamp','user','item'], kind='stable').drop_duplicates(['user','item'], keep='first').reset_index(drop=True)
    assert 100_000 <= len(clean) <= 2_000_000
    assert clean.user.nunique() >= 1000 and clean.item.nunique() >= 1000
    wanted = set(clean.item)
    metadata = {}
    with gzip.open(Path(directory)/'meta_Digital_Music.json.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r.get('asin') in wanted and r['asin'] not in metadata:
                metadata[r['asin']] = {k:r.get(k) for k in ['title','category','brand']}
    return raw, clean, metadata

def split(clean):
    t1, t2 = clean.timestamp.quantile([.8,.9]).to_numpy().astype(int)
    # Ties on a calendar day stay in a single window.
    train = clean[clean.timestamp < t1].copy()
    validation = clean[(clean.timestamp >= t1) & (clean.timestamp < t2)].copy()
    test = clean[clean.timestamp >= t2].copy()
    users = sorted(train.user.unique()); items = sorted(train.item.unique())
    umap = {s:i for i,s in enumerate(users)}; imap = {s:i for i,s in enumerate(items)}
    def encode(frame):
        f = frame[frame.user.isin(umap) & frame.item.isin(imap)].copy()
        f['u'] = f.user.map(umap).astype(int); f['i'] = f.item.map(imap).astype(int)
        return f.reset_index(drop=True)
    audit = []
    for name, f in [('train',train),('validation',validation),('test',test)]:
        audit.append(dict(split=name, raw_events=len(f), warm_events=len(encode(f)),
                          cold_user_events=int((~f.user.isin(umap)).sum()),
                          cold_item_events=int((~f.item.isin(imap)).sum())))
    return encode(train), encode(validation), encode(test), (len(users),len(items)), (t1,t2), pd.DataFrame(audit)

@dataclass
class TrainingData:
    frame: pd.DataFrame
    shape: tuple
    cutoff: int

    def __post_init__(self):
        f = self.frame
        assert f.timestamp.max() < self.cutoff
        self.seen = sparse.csr_matrix((np.ones(len(f)), (f.u,f.i)), shape=self.shape)
        p = f[f.rating >= 4]
        self.positive = sparse.csr_matrix((np.ones(len(p)), (p.u,p.i)), shape=self.shape)
        n_users,n_items = self.shape
        def stats(key, size):
            groups = f.groupby(key)
            counts = groups.size().reindex(range(size),fill_value=0).to_numpy()
            means = groups.rating.mean().reindex(range(size),fill_value=3.).to_numpy()
            return np.minimum(np.floor(np.log2(1+counts)).astype(int),15), np.rint(means).astype(int)
        uc,ur = stats('u',n_users); ic,ir = stats('i',n_items)
        first_format = f.sort_values(['timestamp','user','item'],kind='stable').drop_duplicates('i').set_index('i')['format']
        self.formats = sorted(set(first_format))
        fmap = {s:k for k,s in enumerate(self.formats)}
        fmt = first_format.reindex(range(n_items),fill_value='unknown').map(lambda s:fmap.get(s,len(fmap))).to_numpy()
        last = f.groupby('i').timestamp.max().reindex(range(n_items),fill_value=self.cutoff).to_numpy()
        age = np.digitize((self.cutoff-last)/86400., [7,30,90,365,1095])
        self.user_meta = np.column_stack([uc+n_users+n_items, ur+n_users+n_items+16])
        off = n_users+n_items+22
        self.item_meta = np.column_stack([ic+off, ir+off+16, age+off+22, fmt+off+28])
        self.n_features = off+28+len(fmap)+1

    def features(self, users, items, metadata=True):
        ids = np.column_stack([users, np.asarray(items)+self.shape[0]]).astype(np.int64)
        if metadata:
            return np.column_stack([ids,self.user_meta[users],self.item_meta[items]]).astype(np.int64)
        return ids
