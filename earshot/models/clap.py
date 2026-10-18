"""CLAP (contrastive language-audio pretraining) on CPU: one embedding space for sounds and text.

    from earshot.models import clap
    c = clap.get()
    a = c.embed_file('whoosh.wav')                 # (512,) unit vector, cached on disk
    t = c.embed_text(['a fast air whoosh'])        # (1, 512)
    cos = a @ t.T

Model: laion/larger_clap_general (Apache-2.0) by default, at a pinned revision with safetensors weights, loaded from
earshot's cache only (earshot.models.hub: fetched once when missing, never while an offline flag is set). Its mel
front end covers 50 Hz - 14 kHz at 48 kHz, so it is deaf to "air" above 14 kHz (use earshot.features for bandwidth /
lowpass checks). Inputs longer than 10 s are embedded as deterministic 10 s windows (hop 5 s) and averaged; the
model's own random crop is never used, so results are reproducible. Short sounds are repeat-padded to 10 s like in
CLAP's training (pad='repeatpad'); pad='zero' pads with silence instead (better for judging a single hit's character
in isolation).
"""
import hashlib
import os
import threading

import numpy as np

from .. import audio, config
from . import hub

WIN_S = 10.0
HOP_S = 5.0
_lock = threading.Lock()
_inst = None


def _slug(model_id):
    return hub.slug(model_id)


def _read_npz(path):
    """dict of arrays from an .npz, closing the file (an open NpzFile blocks os.replace on Windows)"""
    if not os.path.exists(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return {}


def _quiet_transformers():
    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        tlog.disable_progress_bar()
    except Exception:  # noqa: BLE001
        pass


def _load_hf(cls, spec, weights=False):
    """a model (weights=True) or its processor from the spec's local snapshot folder (hub.Spec). transformers gets the
    folder, not the repo id, so nothing asks the Hub. A failed load is an error that names the folder, never a
    silent new download; only a legacy snapshot or a local model folder loads pickle weights (pytorch_model.bin,
    torch weights_only)"""
    import warnings
    _quiet_transformers()
    kw = dict(local_files_only=True)
    if weights:
        kw['use_safetensors'] = spec.weights not in hub.PICKLE
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*mel filter.*')
        warnings.filterwarnings('ignore', category=FutureWarning)
        try:
            return cls.from_pretrained(spec.snapshot, **kw)
        except Exception as e:
            if spec.folder:   # the user's own folder: never suggest deleting it
                raise RuntimeError(f'could not load {spec.label} ({cls.__name__}): {type(e).__name__}: {e}. The '
                                   'files may be damaged or this transformers version cannot read them') from e
            raise RuntimeError(f'could not load {spec.label} ({cls.__name__}) from the earshot cache '
                               f'({spec.snapshot}): {type(e).__name__}: {e}. The files may be damaged or this '
                               'transformers version cannot read them: delete that folder and earshot fetches the '
                               'pinned files again, or run earshot doctor') from e


def native_bf16(torch):
    """True when the CPU has native bf16 matrix instructions (AVX512-BF16 or AMX). Without them torch still runs
    bf16, but emulated: measured 16x slower than fp32 on an AVX2-only CPU"""
    try:  # kernels capped below AVX512 (ATEN_CPU_CAPABILITY / ONEDNN_MAX_CPU_ISA) cannot use the bf16 instructions
        if torch.backends.cpu.get_cpu_capability() in ('DEFAULT', 'NO AVX', 'AVX', 'AVX2', 'ZVECTOR', 'VSX'):
            return False
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get('ONEDNN_MAX_CPU_ISA', '').upper() in ('SSE41', 'AVX', 'AVX2', 'AVX2_VNNI', 'AVX512_CORE',
                                                            'AVX512_CORE_VNNI'):
        return False
    cpu = getattr(torch, 'cpu', None)
    for name in ('_is_avx512_bf16_supported', '_is_amx_tile_supported'):
        try:
            if getattr(cpu, name)():
                return True
        except Exception:  # noqa: BLE001  (older torch: no probe -> assume no native bf16)
            pass
    return False


def _set_precision(model, torch):
    """bf16 weights when the CPU has native bf16 (AVX512-BF16 / AMX; on the Zen 4 dev laptop 1.4x faster than fp32,
    embeddings within cosine 0.9993 of it: eval/semantics/precision.json); fp32 otherwise, and always with
    EARSHOT_PRECISION=fp32"""
    if config.PRECISION != 'bf16' or not native_bf16(torch):
        return torch.float32
    try:
        model.to(torch.bfloat16)
        with torch.inference_mode():
            torch.ones(2, 2, dtype=torch.bfloat16) @ torch.ones(2, 2, dtype=torch.bfloat16)
        return torch.bfloat16
    except Exception:
        model.to(torch.float32)
        return torch.float32


class Clap:
    def __init__(self, model_id=None):
        config.ensure_dirs()
        self.model_id = model_id or config.CLAP_MODEL
        self.spec = hub.ensure(self.model_id, config.CLAP_REVISION if self.model_id == config.CLAP_MODEL else None)
        self.revision, self.weights = self.spec.sha, self.spec.weights   # what actually loads (doctor reports it)
        import torch
        from transformers import ClapModel, ClapProcessor
        torch.set_num_threads(config.threads())
        self.torch = torch
        self.model = _load_hf(ClapModel, self.spec, weights=True).eval()
        self.proc = _load_hf(ClapProcessor, self.spec)
        self.dtype = _set_precision(self.model, torch)
        self.precision = 'bf16' if self.dtype == torch.bfloat16 else 'fp32'
        self.dir = os.path.join(config.EMB_CACHE, 'clap', self.spec.cache_name + '--' + self.precision)
        os.makedirs(self.dir, exist_ok=True)
        self._text = {}
        self._text_path = os.path.join(self.dir, 'text.npz')
        self._text = _read_npz(self._text_path)
        self._text_dirty = False
        try:
            self.logit_scale = float(self.model.logit_scale_a.exp().item())
        except Exception:
            self.logit_scale = 100.0

    # ------------------------------------------------------------------ text
    @staticmethod
    def _tkey(t):
        return hashlib.sha1(t.encode('utf-8')).hexdigest()[:24]

    def embed_text(self, texts, batch=64):
        """(n, 512) unit vectors; cached per model on disk"""
        if isinstance(texts, str):
            texts = [texts]
        keys = [self._tkey(t) for t in texts]
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._text]
        torch = self.torch
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            inp = self.proc(text=[t for _, t in chunk], return_tensors='pt', padding=True)
            with torch.inference_mode():
                e = self.model.get_text_features(**inp)
                e = getattr(e, 'pooler_output', e)
            e = e.float().numpy().astype(np.float32)
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
            for (k, _), v in zip(chunk, e):
                self._text[k] = v
            self._text_dirty = True
        if self._text_dirty:
            self._save_text()
        return np.stack([self._text[k] for k in keys]) if keys else np.zeros((0, 512), np.float32)

    def _save_text(self):
        """merge with what other processes saved meanwhile, then replace atomically"""
        import time
        with _lock:
            merged = _read_npz(self._text_path)
            merged.update(self._text)
            self._text = merged
            tmp = self._text_path + f'.{os.getpid()}.tmp.npz'
            np.savez(tmp, **merged)
            for attempt in range(5):  # Windows: another process may hold the file for a moment
                try:
                    os.replace(tmp, self._text_path)
                    break
                except OSError:
                    time.sleep(0.05 * (attempt + 1))
            else:  # a cache write must never fail the call: keep the in-memory cache
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            self._text_dirty = False

    # ------------------------------------------------------------------ audio
    def _prep(self, x, pad):
        x = np.asarray(x, np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        n = int(WIN_S * audio.SR)
        if len(x) < 480:  # < 10 ms: pad so the front end has something to read
            x = np.pad(x, (0, 480 - len(x)))
        if len(x) < n and pad == 'zero':
            x = np.pad(x, (0, n - len(x)))
        return x[:n]

    def _forward(self, segs, pad, batch):
        torch = self.torch
        out = []
        for i in range(0, len(segs), batch):
            chunk = [self._prep(s, pad) for s in segs[i:i + batch]]
            inp = self.proc(audio=chunk, sampling_rate=audio.SR, return_tensors='pt')
            inp = {k: (v.to(self.dtype) if v.is_floating_point() else v) for k, v in inp.items()}
            with torch.inference_mode():
                e = self.model.get_audio_features(**inp)
                e = getattr(e, 'pooler_output', e)
            e = e.float().numpy().astype(np.float32)
            out.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def embed_audio(self, arrays, pad='repeatpad', batch=8, per_window=False):
        """mono (or stereo) 48 kHz arrays -> (n, 512) unit vectors. Arrays longer than 10 s are split into 10 s
        windows (hop 5 s) and averaged; per_window=True returns a list of (k_i, 512) arrays instead."""
        if isinstance(arrays, np.ndarray) and arrays.ndim == 1:
            arrays = [arrays]
        segs, owner = [], []
        n = int(WIN_S * audio.SR)
        for j, x in enumerate(arrays):
            x = audio.to_mono(np.asarray(x, np.float32))
            if len(x) <= n:
                segs.append(x); owner.append(j)
            else:
                for _, w in audio.windows(x, WIN_S, HOP_S, pad_last=True):
                    segs.append(w); owner.append(j)
        e = self._forward(segs, pad, batch)
        owner = np.array(owner)
        groups = [e[owner == j] for j in range(len(arrays))]
        if per_window:
            return groups
        m = np.stack([g.mean(axis=0) for g in groups])
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)

    def embed_file(self, path, start=None, end=None, pad='repeatpad'):
        """(512,) for a file or a [start, end) segment of it; cached on disk"""
        k = hashlib.sha1(f'{audio.file_id(path)}|{start}|{end}|{pad}'.encode()).hexdigest()[:24]
        f = os.path.join(self.dir, 'a_' + k + '.npy')
        if os.path.exists(f):
            try:
                return np.load(f)
            except Exception:
                pass
        x = audio.load(path, mono=True, start=start, end=end)
        e = self.embed_audio([x], pad=pad)[0]
        tmp = f'{f}.{os.getpid()}.tmp.npy'
        np.save(tmp, e)
        audio.replace_quietly(tmp, f)
        return e

    def embed_files(self, paths, pad='repeatpad', batch=8):
        """(n, 512) for many files: cached ones are read, the rest are embedded in batches"""
        out = [None] * len(paths)
        todo = []
        for i, p in enumerate(paths):
            k = hashlib.sha1(f'{audio.file_id(p)}|None|None|{pad}'.encode()).hexdigest()[:24]
            f = os.path.join(self.dir, 'a_' + k + '.npy')
            if os.path.exists(f):
                try:
                    out[i] = np.load(f)
                    continue
                except Exception:
                    pass
            todo.append((i, p, f))
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            xs = [audio.load(p, mono=True) for _, p, _ in chunk]
            es = self.embed_audio(xs, pad=pad, batch=batch)
            for (i, _, f), e in zip(chunk, es):
                out[i] = e
                tmp = f'{f}.{os.getpid()}.tmp.npy'
                np.save(tmp, e)
                audio.replace_quietly(tmp, f)
        return np.stack(out) if out else np.zeros((0, 512), np.float32)


def get(model_id=None):
    """process-wide singleton (the MCP server keeps it warm between calls)"""
    global _inst
    with _lock:
        if _inst is None or (model_id and _inst.model_id != model_id):
            _inst = Clap(model_id)
        return _inst
