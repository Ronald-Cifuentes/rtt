/**
 * AudioWorklet processor for capturing microphone audio.
 * Converts float32 samples to PCM16 and sends them to the main thread.
 * Runs at 16kHz mono (resampled from AudioContext's native sample rate).
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetSampleRate = options.processorOptions?.targetSampleRate || 16000;
    this.bufferSize = options.processorOptions?.bufferSize || 1600; // 100ms at 16kHz
    this.buffer = new Float32Array(0);
    this.isActive = true;

    this.port.onmessage = (event) => {
      if (event.data.type === 'stop') {
        this.isActive = false;
      } else if (event.data.type === 'start') {
        this.isActive = true;
      }
    };
  }

  process(inputs, outputs, parameters) {
    if (!this.isActive) return true;

    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;

    const inputData = input[0]; // mono channel

    // Resample from native rate to target rate
    const ratio = this.targetSampleRate / sampleRate;
    const resampled = this._resample(inputData, ratio);

    // Accumulate in buffer
    const newBuffer = new Float32Array(this.buffer.length + resampled.length);
    newBuffer.set(this.buffer);
    newBuffer.set(resampled, this.buffer.length);
    this.buffer = newBuffer;

    // Send chunks of bufferSize
    while (this.buffer.length >= this.bufferSize) {
      const chunk = this.buffer.slice(0, this.bufferSize);
      this.buffer = this.buffer.slice(this.bufferSize);

      // Convert float32 to PCM16
      const pcm16 = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      this.port.postMessage(
        { type: 'audio', pcm16: pcm16.buffer },
        [pcm16.buffer]
      );
    }

    return true;
  }

  _resample(inputData, ratio) {
    if (ratio === 1) return inputData;

    const outputLength = Math.round(inputData.length * ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i / ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, inputData.length - 1);
      const frac = srcIndex - srcIndexFloor;
      output[i] = inputData[srcIndexFloor] * (1 - frac) + inputData[srcIndexCeil] * frac;
    }

    return output;
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
