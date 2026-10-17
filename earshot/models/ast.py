"""AudioSet tagger (Audio Spectrogram Transformer) on CPU: 527 sound classes with independent probabilities.

    from earshot.models import ast
    t = ast.get()
    t.top_file('beep.wav', k=5)   # [('Beep, bleep', 0.71), ('Sine wave', 0.33), ...]

A second, independent "ear" next to CLAP: it was trained supervised on AudioSet, so its probabilities are calibrated
per class (CLAP's cosine scores are not). Useful classes for film sound: 'Synthesizer', 'Sine wave', 'Beep, bleep',
'Whoosh, swoosh, swish', 'Typing', 'Computer keyboard', 'Click', 'Boom', 'Music', 'Speech', 'Silence' ...
Input is resampled to 16 kHz (so it hears up to 8 kHz only) and read in 10.24 s windows; longer audio is split into
windows and the per-class maximum (or mean) is taken.
"""
import hashlib
import os
import threading
import warnings

import numpy as np

from .. import audio, config
from . import hub
from .clap import _load_hf, _set_precision

SR_AST = 16000
WIN_S = 10.24
_lock = threading.Lock()
_inst = None


class AST:
    def __init__(self, model_id=None):
        config.ensure_dirs()
        self.model_id = model_id or config.AST_MODEL
        self.spec = hub.ensure(self.model_id, config.AST_REVISION if self.model_id == config.AST_MODEL else None)
        self.revision, self.weights = self.spec.sha, self.spec.weights
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        torch.set_num_threads(config.threads())
        self.torch = torch
        self.model = _load_hf(ASTForAudioClassification, self.spec, weights=True).eval()
        self.fe = _load_hf(ASTFeatureExtractor, self.spec)
        self.dtype = _set_precision(self.model, torch)
        self.precision = 'bf16' if self.dtype == torch.bfloat16 else 'fp32'
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        self.index = {l: i for i, l in enumerate(self.labels)}
        self.dir = os.path.join(config.EMB_CACHE, 'ast', self.spec.cache_name + '--' + self.precision)
        os.makedirs(self.dir, exist_ok=True)

    def _forward(self, segs16, batch=8):
        torch = self.torch
        out = []
        for i in range(0, len(segs16), batch):
            chunk = [np.pad(s, (0, max(0, 1600 - len(s)))) for s in segs16[i:i + batch]]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*mel filter.*')
                inp = self.fe(chunk, sampling_rate=SR_AST, return_tensors='pt')
            inp = {k: (v.to(self.dtype) if v.is_floating_point() else v) for k, v in inp.items()}
            with torch.inference_mode():
                logits = self.model(**inp).logits
            out.append(torch.sigmoid(logits.float()).numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), np.float32)

    def probs(self, arrays, reduce='max', batch=8):
        """48 kHz arrays -> (n, 527) class probabilities (windows of 10.24 s reduced by max or mean)"""
        if isinstance(arrays, np.ndarray) and arrays.ndim == 1:
            arrays = [arrays]
        segs, owner = [], []
        n = int(WIN_S * SR_AST)
        for j, x in enumerate(arrays):
            y = audio.resample(audio.to_mono(np.asarray(x, np.float32)), audio.SR, SR_AST)
            if len(y) <= n:
                segs.append(y); owner.append(j)
            else:
                for s in range(0, len(y) - n + 1, n // 2):
                    segs.append(y[s:s + n]); owner.append(j)
                if (len(y) - n) % (n // 2):
                    segs.append(y[-n:]); owner.append(j)
        p = self._forward(segs, batch)
        owner = np.array(owner)
        red = np.max if reduce == 'max' else np.mean
        return np.stack([red(p[owner == j], axis=0) for j in range(len(arrays))])

    def probs_file(self, path, start=None, end=None, reduce='max'):
        k = hashlib.sha1(f'{audio.file_id(path)}|{start}|{end}|{reduce}'.encode()).hexdigest()[:24]
        f = os.path.join(self.dir, 'p_' + k + '.npy')
        if os.path.exists(f):
            try:
                return np.load(f)
            except Exception:
                pass
        p = self.probs([audio.load(path, mono=True, start=start, end=end)], reduce=reduce)[0]
        tmp = f'{f}.{os.getpid()}.tmp.npy'
        np.save(tmp, p)
        audio.replace_quietly(tmp, f)
        return p

    def top(self, p, k=10, min_p=0.0):
        idx = np.argsort(p)[::-1][:k]
        return [(self.labels[i], round(float(p[i]), 3)) for i in idx if p[i] >= min_p]

    def top_file(self, path, k=10, start=None, end=None):
        return self.top(self.probs_file(path, start, end), k)

    def prob_of(self, p, label):
        return float(p[self.index[label]])


def get(model_id=None):
    global _inst
    with _lock:
        if _inst is None or (model_id and _inst.model_id != model_id):
            _inst = AST(model_id)
        return _inst
