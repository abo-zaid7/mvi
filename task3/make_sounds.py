"""
Generate the GUI's sound effects.

Run once with:   python make_sounds.py

The brief allows "playsound or other library and available/free sound effects".
Rather than download clips with unclear licences, the six cues are synthesised
here with numpy and written as WAV files, so they are unambiguously free to
redistribute with the assignment and the repository stays small (about 30 kB for
all six).

The cues are deliberately short and deliberately different from each other in
pitch direction, because that is what makes them distinguishable without being
looked at:

    ready         two rising notes    - the app is ready
    model_loaded  three rising notes  - a model finished loading
    predict_done  a soft major chord  - a prediction finished
    error         two falling notes   - something went wrong
    prompt        one short blip      - an input is needed
    click         a very short tick   - a control was used

Falling pitch for errors and rising pitch for success is not decoration: the
association is consistent across listeners and means the meaning survives even
when the sound is heard at the edge of hearing or through poor speakers.
"""

import os
import wave

import numpy as np

RATE = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(HERE, "static", "sounds")


def tone(frequency, seconds, volume=0.35, harmonic=0.25):
    """
    One note.

    A single sine wave sounds thin and buzzy on laptop speakers, so a quieter
    octave above is mixed in, and both ends are faded to avoid the click you get
    from a waveform that starts or stops mid-cycle.
    """
    t = np.linspace(0, seconds, int(RATE * seconds), endpoint=False)
    wave_data = np.sin(2 * np.pi * frequency * t)
    wave_data += harmonic * np.sin(4 * np.pi * frequency * t)

    fade = int(RATE * 0.012)
    envelope = np.ones_like(wave_data)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)

    # A gentle decay over the whole note stops it sounding like a test signal.
    envelope *= np.linspace(1.0, 0.55, len(wave_data))

    return volume * wave_data * envelope


def chord(frequencies, seconds, volume=0.30):
    """Several notes at once, used for the "finished" cue."""
    mixed = sum(tone(f, seconds, volume=1.0) for f in frequencies)
    return volume * mixed / len(frequencies)


def silence(seconds):
    return np.zeros(int(RATE * seconds))


def write(name, samples):
    """Write one mono 16-bit WAV file."""
    os.makedirs(SOUND_DIR, exist_ok=True)
    path = os.path.join(SOUND_DIR, name + ".wav")

    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / max(peak, 1.0)          # only ever quieten, never boost
    clipped = np.clip(samples, -1.0, 1.0)
    frames = (clipped * 32767).astype(np.int16)

    with wave.open(path, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(frames.tobytes())

    print("  %-14s %5.2f s  %5d bytes" % (name, len(samples) / RATE,
                                          os.path.getsize(path)))


# Note frequencies used below, so the sequences read as music rather than numbers.
A4, C5, D5, E5, F5, G5, A5, C6, E6 = 440, 523, 587, 659, 698, 784, 880, 1047, 1319
E4, G4 = 330, 392


def main():
    print("writing sound effects to", SOUND_DIR)

    # The app is ready to use.
    write("ready", np.concatenate([tone(E5, 0.09), tone(A5, 0.16)]))

    # A model has finished loading - three rising notes, the most "arrived" cue.
    write("model_loaded", np.concatenate([
        tone(C5, 0.08), tone(E5, 0.08), tone(G5, 0.20)]))

    # A prediction has finished - a soft chord, distinctly not a single note.
    write("predict_done", np.concatenate([
        chord([C5, E5, G5], 0.13), chord([E5, G5, C6], 0.22)]))

    # Something went wrong - falling, and lower than every other cue.
    write("error", np.concatenate([
        tone(G4, 0.13, volume=0.40), silence(0.03), tone(E4, 0.26, volume=0.40)]))

    # An input is needed from the user.
    write("prompt", np.concatenate([tone(A5, 0.07, volume=0.28),
                                    silence(0.04),
                                    tone(A5, 0.07, volume=0.28)]))

    # A control was used - as short as it can be and still be heard.
    write("click", tone(E6, 0.035, volume=0.16, harmonic=0.0))

    print("done - %d files" % len(os.listdir(SOUND_DIR)))


if __name__ == "__main__":
    main()
