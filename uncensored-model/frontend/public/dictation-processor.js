const MAX_SECONDS = 120;
const TARGET_RATE = 16000;

class DictationProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = new Int16Array(TARGET_RATE * MAX_SECONDS);
    this.length = 0;
    this.sampleSpace = sampleRate;
    this.sampleSum = 0;
    this.finished = false;
    this.port.onmessage = () => this.finish();
  }

  finish() {
    if (this.finished) return;
    this.finished = true;
    this.port.postMessage({ buffer: this.samples.buffer, length: this.length }, [this.samples.buffer]);
  }

  process(inputs) {
    if (this.finished) return false;
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    // Average the native-rate samples that overlap each 16 kHz output sample.
    for (const input of channel) {
      let available = TARGET_RATE;
      while (available > 0) {
        const overlap = Math.min(available, this.sampleSpace);
        this.sampleSum += input * overlap;
        this.sampleSpace -= overlap;
        available -= overlap;
        if (this.sampleSpace === 0) {
          const sample = Math.max(-1, Math.min(1, this.sampleSum / sampleRate));
          this.samples[this.length++] = sample * (sample < 0 ? 32768 : 32767);
          if (this.length === this.samples.length) {
            this.finish();
            return false;
          }
          this.sampleSpace = sampleRate;
          this.sampleSum = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('abyss-dictation', DictationProcessor);
