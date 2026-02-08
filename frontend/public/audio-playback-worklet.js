/**
 * AudioWorklet processor for smooth audio playback.
 * Receives PCM16 chunks from the main thread and plays them without gaps.
 * Outputs at 24kHz (Qwen3-TTS native rate).
 */
class AudioPlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.sourceSampleRate = options.processorOptions?.sourceSampleRate || 24000;
    this.ringBufferSize = 48000 * 5; // 5 seconds buffer at 48kHz
    this.ringBuffer = new Float32Array(this.ringBufferSize);
    this.writePos = 0;
    this.readPos = 0;
    this.bufferedSamples = 0;
    this.isPlaying = false;
    this.minBufferBeforePlay = 2400; // ~50ms at 48kHz before starting playback

    this.port.onmessage = (event) => {
      if (event.data.type === 'audio') {
        this._addAudioData(event.data.pcm16);
      } else if (event.data.type === 'clear') {
        this._clear();
      }
    };
  }

  _addAudioData(pcm16ArrayBuffer) {
    const pcm16 = new Int16Array(pcm16ArrayBuffer);
    // Convert PCM16 to float32
    const float32 = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) {
      float32[i] = pcm16[i] / 32768.0;
    }

    // Resample from source rate (24kHz) to output rate (48kHz typically)
    const ratio = sampleRate / this.sourceSampleRate;
    const resampled = this._resample(float32, ratio);

    // Write to ring buffer
    for (let i = 0; i < resampled.length; i++) {
      this.ringBuffer[this.writePos] = resampled[i];
      this.writePos = (this.writePos + 1) % this.ringBufferSize;
      this.bufferedSamples++;
    }

    // Start playing once we have enough buffered
    if (!this.isPlaying && this.bufferedSamples >= this.minBufferBeforePlay) {
      this.isPlaying = true;
    }

    // Report buffer level to main thread
    this.port.postMessage({
      type: 'bufferLevel',
      samples: this.bufferedSamples
    });
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const outputChannel = output[0];

    if (!this.isPlaying || this.bufferedSamples <= 0) {
      // Output silence
      outputChannel.fill(0);
      if (this.isPlaying && this.bufferedSamples <= 0) {
        this.isPlaying = false;
        this.port.postMessage({ type: 'playbackEnded' });
      }
      return true;
    }

    for (let i = 0; i < outputChannel.length; i++) {
      if (this.bufferedSamples > 0) {
        outputChannel[i] = this.ringBuffer[this.readPos];
        this.readPos = (this.readPos + 1) % this.ringBufferSize;
        this.bufferedSamples--;
      } else {
        outputChannel[i] = 0;
      }
    }

    return true;
  }

  _resample(inputData, ratio) {
    if (Math.abs(ratio - 1) < 0.001) return inputData;

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

  _clear() {
    this.ringBuffer.fill(0);
    this.writePos = 0;
    this.readPos = 0;
    this.bufferedSamples = 0;
    this.isPlaying = false;
  }
}

registerProcessor('audio-playback-processor', AudioPlaybackProcessor);
