import threading
import numpy as np
import sounddevice as sd


class ThroatVibrationSensor:
    """
    STATUS: simulated. We're reading the laptop mic, not a piezo throat
    sensor -- say this explicitly in the demo (per the honesty framing in
    the brief). If you get real piezo+Arduino hardware working, replace the
    body of read()/the audio callback with serial reads from the board and
    keep the same 0-1 'confidence' contract so app.py doesn't need to change.

    Heuristic: voiced/nasal sounds (humming "m", buzzing "b") carry more
    low-frequency energy than pure breath/plosive sounds ("p", whisper).
    We combine a loudness gate with a low-band-vs-total FFT energy ratio
    into one 0-1 confidence score. This is a hand-tuned proxy, not a
    trained classifier -- swap resolve_ambiguous() in fusion.py for a real
    model once you have labeled examples.
    """

    def __init__(self, samplerate=16000, block_size=4096, low_band_hz=300,
                 rms_gate=0.02, device=None, smoothing=0.6):
        """
        block_size=4096 (~256ms at 16kHz) instead of 1024 (~64ms) -- the
        shorter window was too noisy frame-to-frame, which is most of what
        looked like "random" jumping around.

        device: sounddevice input device index. Leave None to use the
        system default, or set it explicitly if the wrong mic gets picked
        (see list_audio_devices.py to find the right index).

        smoothing: 0-1 exponential moving average factor applied to the
        confidence value. Higher = smoother/slower to react, lower = more
        responsive but jumpier. 0.6 is a reasonable starting point.
        """
        self.samplerate = samplerate
        self.block_size = block_size
        self.low_band_hz = low_band_hz
        self.rms_gate = rms_gate  # tune this to your mic's noise floor
        self.device = device
        self.smoothing = smoothing

        self._latest_confidence = 0.0
        self._latest_rms = 0.0
        self._lock = threading.Lock()
        self._stream = None
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[vibration_sensor] stream status: {status}", flush=True)

        samples = indata[:, 0]
        rms = float(np.sqrt(np.mean(samples ** 2)))

        fft = np.fft.rfft(samples)
        freqs = np.fft.rfftfreq(len(samples), 1.0 / self.samplerate)
        mag = np.abs(fft)
        total_energy = mag.sum() + 1e-9
        low_energy = mag[freqs <= self.low_band_hz].sum()
        low_ratio = low_energy / total_energy

        loudness_gate = min(rms / self.rms_gate, 1.0)
        raw_confidence = float(np.clip(loudness_gate * low_ratio * 2.0, 0.0, 1.0))

        with self._lock:
            # Exponential smoothing so single noisy blocks don't cause big
            # visible jumps in the UI.
            self._latest_confidence = (
                self.smoothing * self._latest_confidence
                + (1 - self.smoothing) * raw_confidence
            )
            self._latest_rms = rms

    def start(self):
        if self._running:
            return
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.block_size,
            channels=1,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._running = False

    def read(self):
        with self._lock:
            return {"confidence": self._latest_confidence, "rms": self._latest_rms}
