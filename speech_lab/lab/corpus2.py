"""Uniform utterance table over speech corpora (macOS TTS now, LibriSpeech-style corpora later).

Every corpus is a table of `Utt` rows (id, speaker, text, path, duration, pool, source, split, gender,
style) plus a speaker table (pool, split, gender, quality, roles). Generators pick speakers by role:
  'partner'  bus partners,  'talker'  meeting talkers,  'babble'  every bus interferer (passengers,
  conversing pairs, far babble).
Pools are speaker-level and disjoint: 'target' speakers (partner/talker roles) never appear as babble.

Corpora:
  'tts'   built from corpus/manifest.json (see TTS_* below for the voice eligibility decision).
  <name>  read from corpora/<name>/manifest.json (relative to LAB), format in BENCH2.md "Scene format";
          a LibriSpeech drop-in only has to write that manifest.
"""
import functools
import json
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .common import FS, active_rms

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- TTS voice eligibility (DESIGN 4.1: formant voices and novelty voices are not natural speech).
# Natural English voices: bus partners and meeting talkers (22 dialogue + 20 meeting + 8 babble lines each).
TTS_PARTNERS = ["Samantha", "Daniel", "Karen", "Moira", "Tessa", "Rishi", "Aman", "Tara"]
# A meeting needs 20 distinct voices but only 8 natural English voices exist, so 12 natural voices of
# European locales reading English (8 babble lines each) are moved from the babble pool to the target pool.
TTS_TALKERS_ACCENTED = ["Anna", "Thomas", "Xander", "Ellen", "Nora", "Alva", "Sara", "Satu",
                        "Alice", "Mónica", "Joana", "Zosia"]
# Excluded everywhere: MacinTalk formant synthesizers and the Eloquence novelty voices ("Name (English (..))").
TTS_FORMANT = ["Fred", "Kathy", "Ralph", "Junior"]
TTS_GENDER = {"Samantha": "F", "Daniel": "M", "Karen": "F", "Moira": "F", "Tessa": "F", "Rishi": "M",
              "Aman": "M", "Tara": "F", "Thomas": "M", "Xander": "M", "Jacques": "M"}
TTS_STYLE = {"bus": "dialogue", "mtg": "meeting", "bab": "babble"}

DEFAULT_ROLES = {"target": ["partner", "talker"], "babble": ["babble"]}
POOLS = ("target", "babble")
SPLITS = ("dev", "test", "all")


@dataclass(frozen=True)
class Utt:
    id: str
    speaker: str
    text: str
    path: str        # relative to LAB
    duration: float  # seconds
    pool: str        # 'target' | 'babble'
    source: str      # 'tts' | 'libri' | ...
    split: str = "all"   # 'dev' | 'test' | 'all' (usable in every set)
    gender: str = "?"
    style: str = "read"  # TTS: 'dialogue' | 'meeting' | 'babble'


class Corpus:
    """Utterance table plus per-speaker attributes; audio is loaded lazily and cached."""

    def __init__(self, name, source, utts, speakers):
        self.name, self.source = name, source
        self.utts = sorted(utts, key=lambda u: u.id)
        self._by_id = {u.id: u for u in self.utts}
        if len(self._by_id) != len(self.utts):
            raise ValueError(f"corpus {name}: duplicate utterance ids")
        self.speakers = {}
        for spk, info in speakers.items():
            us = [u for u in self.utts if u.speaker == spk]
            if not us:
                continue
            if info["pool"] not in POOLS or info.get("split", "all") not in SPLITS:
                raise ValueError(f"corpus {name}: bad pool/split for speaker {spk}: {info}")
            self.speakers[spk] = dict(pool=info["pool"], split=info.get("split", "all"),
                                      gender=info.get("gender", "?"), quality=info.get("quality", "natural"),
                                      roles=list(info.get("roles", DEFAULT_ROLES[info["pool"]])),
                                      n_utts=len(us), total_s=round(sum(u.duration for u in us), 3))
        missing = {u.speaker for u in self.utts} - set(self.speakers)
        if missing:
            raise ValueError(f"corpus {name}: utterances of unknown speakers {sorted(missing)[:5]}")

    def __contains__(self, utt_id):
        return utt_id in self._by_id

    def utt(self, utt_id):
        return self._by_id[utt_id]

    def select(self, speaker=None, pool=None, split=None, role=None, style=None, min_dur=0.0, max_dur=np.inf):
        """Utterances matching all given filters (split='dev' also matches split 'all')."""
        out = []
        for u in self.utts:
            sp = self.speakers[u.speaker]
            if ((speaker is None or u.speaker == speaker) and (pool is None or u.pool == pool)
                    and (split is None or u.split in (split, "all")) and (role is None or role in sp["roles"])
                    and (style is None or u.style == style) and min_dur <= u.duration <= max_dur):
                out.append(u)
        return out

    def speakers_for(self, role, split=None):
        """Sorted speaker ids eligible for `role` ('partner' | 'talker' | 'babble') in `split`."""
        return sorted(s for s, sp in self.speakers.items()
                      if role in sp["roles"] and (split is None or sp["split"] in (split, "all")))

    def audio(self, utt_id):
        """Mono float32 at 16 kHz, scaled to unit active RMS (lab.common.active_rms). Read-only, cached."""
        return _load_norm(os.path.join(LAB, self._by_id[utt_id].path))


@functools.lru_cache(maxsize=512)  # about 2-3 scenes of utterances; bounds memory for long corpora
def _load_norm(path):
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    if fs != FS:
        raise ValueError(f"{path}: fs {fs} != {FS}")
    x = x[:, 0].astype(np.float64)
    x = (x / (active_rms(x) + 1e-12)).astype(np.float32)
    x.flags.writeable = False
    return x


def _tts_speaker(voice):
    """Speaker attributes of one TTS voice, or None if excluded."""
    if voice in TTS_FORMANT or "(English (" in voice:
        return None
    g = TTS_GENDER.get(voice, "F")
    if voice in TTS_PARTNERS:
        return dict(pool="target", gender=g, quality="natural", roles=["partner", "talker"])
    if voice in TTS_TALKERS_ACCENTED:
        return dict(pool="target", gender=g, quality="accented", roles=["talker"])
    return dict(pool="babble", gender=g, quality="accented", roles=["babble"])


def _build_tts():
    man = json.load(open(os.path.join(LAB, "corpus", "manifest.json")))
    speakers, utts = {}, []
    for e in man:
        sp = _tts_speaker(e["voice"])
        if sp is None:
            continue
        speakers[e["voice"]] = sp
        path = os.path.join("corpus", e["file"])
        info = sf.info(os.path.join(LAB, path))
        stem = os.path.splitext(e["file"])[0]
        utts.append(Utt(id=stem, speaker=e["voice"], text=e["text"], path=path,
                        duration=info.frames / info.samplerate, pool=sp["pool"], source="tts",
                        gender=sp["gender"], style=TTS_STYLE[stem.split("__")[1][:3]]))
    return Corpus("tts", "tts", utts, speakers)


def _build_manifest(name, path):
    m = json.load(open(path))
    spk = m["speakers"]
    utts = [Utt(id=u["id"], speaker=u["speaker"], text=u.get("text", ""), path=u["path"],
                duration=float(u["duration"]), pool=spk[u["speaker"]]["pool"], source=m.get("source", name),
                split=spk[u["speaker"]].get("split", "all"), gender=spk[u["speaker"]].get("gender", "?"),
                style=u.get("style", "read")) for u in m["utterances"]]
    return Corpus(name, m.get("source", name), utts, spk)


_REGISTERED = {}


def register(name, manifest_path):
    """Use `manifest_path` for corpus `name` in this process (tests; normal corpora live in corpora/)."""
    _REGISTERED[name] = manifest_path
    load_corpus.cache_clear()


@functools.lru_cache(maxsize=8)
def load_corpus(name="tts"):
    """The Corpus called `name` ('tts' or a manifest under corpora/<name>/manifest.json)."""
    if name == "tts":
        return _build_tts()
    path = _REGISTERED.get(name, os.path.join(LAB, "corpora", name, "manifest.json"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"corpus '{name}': no manifest at {path}")
    return _build_manifest(name, path)
