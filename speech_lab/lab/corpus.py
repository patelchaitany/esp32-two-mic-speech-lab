"""Render test speech with macOS `say` voices (16 kHz mono). Original sentences, no dataset download."""
import os, subprocess, re, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import FS, read_wav, write_wav, active_rms, save_json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "corpus")

BUS_DIALOGUE = [
    "Did you finish the report for tomorrow's review?",
    "Almost. I still need to add the charts from last week.",
    "The bus is packed today. I could barely find a seat.",
    "It's always like this on Mondays, everyone goes back to the office.",
    "Are you coming to the team dinner on Friday evening?",
    "I think so, but I have to leave early to catch the last train.",
    "My sister is visiting next month, so I'm looking for a good restaurant.",
    "Try the small Italian place near the station. Their pasta is excellent.",
    "Did you hear that the library is closing for renovations?",
    "Yes, they said it will reopen in the spring with a new cafe.",
    "I started learning the guitar, but my fingers hurt after ten minutes.",
    "That gets better after a couple of weeks. Just keep practicing every day.",
    "What time does your meeting start this afternoon?",
    "At three, and it will probably run late as usual.",
    "We should plan a hiking trip before the weather gets cold.",
    "Good idea. The trail by the lake has beautiful views in October.",
    "I forgot my umbrella again, and the forecast says heavy rain.",
    "You can borrow mine. I keep a spare one in my bag.",
    "How is the new apartment? Is it quieter than the old one?",
    "Much quieter, although the neighbors have a very loud dog.",
    "Should we get off one stop early and walk through the park?",
    "Sure, I could use the fresh air before work.",
]

MEETING = [
    "Good morning everyone. Let's start with a quick update from each team.",
    "The mobile release is on track, and testing finishes on Thursday.",
    "We found two critical bugs in the payment flow that need attention.",
    "Marketing will launch the new campaign at the end of the month.",
    "Our customer satisfaction score went up by four points this quarter.",
    "I need two more engineers to finish the data migration in time.",
    "The new office in Berlin opens in March, and hiring has already started.",
    "Can we move the design review to Wednesday afternoon?",
    "Server costs dropped by twelve percent after the latest optimization.",
    "Sales closed three large contracts last week, mostly in healthcare.",
    "The legal team approved the updated privacy policy yesterday.",
    "We should collect more feedback from users before changing the dashboard.",
    "Training sessions for the new tools will run every Tuesday morning.",
    "The security audit results are positive, with only minor recommendations.",
    "Support ticket volume is down, but response times are still too long.",
    "The partner integration is delayed because their interface is not ready.",
    "Finance needs everyone's budget estimates by the fifteenth.",
    "I suggest we try a shorter sprint for the next cycle.",
    "Hardware samples arrived, and the first tests look promising.",
    "Thanks everyone. I will send the notes and action items after this meeting.",
]

BABBLE = [
    "Excuse me, is this seat taken?", "Next stop is Central Square.",
    "I can't believe how expensive groceries have become.", "Did you watch the game last night?",
    "My phone battery is almost dead.", "We need to buy a present for her birthday.",
    "The traffic on the bridge was terrible this morning.", "I'll call you when I get home.",
    "Could you move your bag, please?", "The kids have a school trip next week.",
    "I think we missed our connection.", "This song has been stuck in my head all day.",
    "Let me know if you want to grab lunch.", "The new bakery downtown is amazing.",
    "He said the package would arrive on Tuesday.", "We should have taken the earlier bus.",
    "My manager wants the slides by noon.", "It's colder than yesterday, isn't it?",
    "I'm reading a great book about the ocean.", "They are renovating the old theater.",
    "Do you know where the pharmacy is?", "I left my keys at the office again.",
    "The dentist appointment is on Thursday.", "We're thinking about getting a cat.",
    "That movie was much longer than I expected.", "Can you believe it's already November?",
    "The price of tickets went up again.", "I need to stop at the bank before five.",
    "My brother is moving to another city.", "Have you tried the noodle shop on Fifth?",
    "I think it's going to snow this weekend.", "Our flight leaves at six in the morning.",
]

TARGET_VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Tessa", "Rishi", "Aman", "Tara",
                 "Fred", "Kathy", "Ralph", "Junior",
                 "Eddy (English (US))", "Flo (English (US))", "Grandma (English (US))", "Grandpa (English (US))",
                 "Reed (English (US))", "Rocko (English (US))", "Sandy (English (US))", "Shelley (English (US))",
                 "Eddy (English (UK))", "Flo (English (UK))", "Reed (English (UK))", "Sandy (English (UK))"]
BABBLE_VOICES = ["Anna", "Thomas", "Jacques", "Amélie", "Alice", "Mónica", "Paulina", "Luciana", "Joana",
                 "Xander", "Ellen", "Nora", "Alva", "Sara", "Satu", "Zosia", "Milena", "Yelda", "Melina",
                 "Ioana", "Laura", "Tina", "Lana", "Tünde", "Zuzana", "Daria", "Lesya", "Ona", "Carmit",
                 "Lekha", "Kyoko", "Yuna", "Tingting", "Meijia", "Sinji", "Linh", "Kanya", "Damayanti", "Amira",
                 "Grandma (English (UK))", "Grandpa (English (UK))", "Rocko (English (UK))", "Shelley (English (UK))"]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def render(voice, text, path, rate=None):
    if os.path.exists(path):
        return path
    tmp = path + ".tmp.wav"
    cmd = ["say", "-v", voice, "-o", tmp, "--file-format=WAVE", f"--data-format=LEI16@{FS}"]
    if rate:
        cmd += ["-r", str(rate)]
    subprocess.run(cmd + [text], check=True, capture_output=True)
    x, fs = read_wav(tmp)
    os.remove(tmp)
    assert fs == FS, fs
    # trim leading/trailing silence, normalize active level to -26 dBFS-ish (scaled again in scenes)
    env = np.abs(x) > 1e-3
    idx = np.where(env)[0]
    x = x[max(idx[0] - 160, 0): idx[-1] + 160] if len(idx) else x
    x = x / (active_rms(x) + 1e-12) * 0.05
    write_wav(path, x)
    return path


def build():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for v in TARGET_VOICES:
        for i, t in enumerate(BUS_DIALOGUE):
            jobs.append((v, t, f"{slug(v)}__bus{i:02d}.wav"))
        for i, t in enumerate(MEETING):
            jobs.append((v, t, f"{slug(v)}__mtg{i:02d}.wav"))
    for k, v in enumerate(BABBLE_VOICES + TARGET_VOICES):
        for j in range(8):  # each babble voice gets 8 sentences
            i = (k * 5 + j * 3) % len(BABBLE)
            jobs.append((v, BABBLE[i], f"{slug(v)}__bab{i:02d}.wav"))
    manifest = []

    def run(job):
        v, t, name = job
        h = int(hashlib.md5(name.encode()).hexdigest(), 16)
        rate = 165 + h % 50  # 165-214 wpm: vary speaking rate
        render(v, t, os.path.join(OUT, name), rate)
        return {"voice": v, "text": t, "file": name}

    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(run, jobs):
            manifest.append(r)
    save_json(os.path.join(OUT, "manifest.json"), manifest)
    print(f"{len(manifest)} utterances in {OUT}")


if __name__ == "__main__":
    build()
