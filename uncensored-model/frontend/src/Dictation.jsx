import React, { useEffect, useRef, useState } from 'react';
import Icon from './Icon';

const SAMPLE_RATE = 16000;

function encodeWav(buffer, length) {
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  new Uint8Array(header).set(new TextEncoder().encode('RIFF'), 0);
  new Uint8Array(header).set(new TextEncoder().encode('WAVEfmt '), 8);
  new Uint8Array(header).set(new TextEncoder().encode('data'), 36);
  view.setUint32(4, 36 + length * 2, true);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  view.setUint32(40, length * 2, true);
  return new Blob([header, new Uint8Array(buffer, 0, length * 2)], { type: 'audio/wav' });
}

function releaseCapture(run) {
  if (run.captureReleased) return;
  run.captureReleased = true;
  clearTimeout(run.stopTimeout);
  run.stream?.getTracks().forEach(track => track.stop());
  run.node?.port.close();
  run.node?.disconnect();
  if (run.context && run.context.state !== 'closed') run.context.close().catch(() => {});
}

function microphoneError(error) {
  if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
    return 'Microphone permission denied. Allow access in your browser to dictate.';
  }
  if (error.name === 'NotFoundError' || error.name === 'OverconstrainedError') return 'No microphone was found.';
  if (error.name === 'NotReadableError' || error.name === 'AbortError') return 'The microphone could not be opened. Check the input device and try again.';
  if (error.name === 'NotSupportedError') return 'This browser does not support microphone capture on the current device.';
  return 'Recording could not start. Check your microphone and browser.';
}

export default function Dictation({ disabled, onTranscript, onActiveChange, onError }) {
  const recording = useRef(null);
  const [phase, setPhase] = useState('idle');
  const [elapsed, setElapsed] = useState(0);

  function cancel() {
    const run = recording.current;
    if (!run) return;
    recording.current = null;
    releaseCapture(run);
    run.request.abort();
    setPhase('idle');
    onActiveChange(false);
  }

  useEffect(() => {
    const leave = () => cancel();
    const hide = () => { if (document.hidden) leave(); };
    window.addEventListener('pagehide', leave);
    document.addEventListener('visibilitychange', hide);
    return () => {
      window.removeEventListener('pagehide', leave);
      document.removeEventListener('visibilitychange', hide);
      leave();
    };
  }, [onActiveChange]);

  useEffect(() => {
    if (disabled) cancel();
  }, [disabled]);

  useEffect(() => {
    if (phase !== 'recording') return;
    const interval = setInterval(() => {
      const run = recording.current;
      if (run) setElapsed(Math.floor((performance.now() - run.started) / 1000));
    }, 250);
    return () => clearInterval(interval);
  }, [phase]);

  async function finish(run, { buffer, length }) {
    if (recording.current !== run) return;
    releaseCapture(run);
    setPhase('processing');
    try {
      if (!length) throw new Error('No microphone audio was captured. Try again.');
      const response = await fetch('/api/transcribe', {
        method: 'POST', body: encodeWav(buffer, length), signal: run.request.signal,
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || typeof result.text !== 'string' || !result.text.trim()) {
        throw new Error(typeof result.detail === 'string' ? result.detail : 'Transcription failed. Try again.');
      }
      if (recording.current === run) onTranscript(result.text.trim());
    } catch (error) {
      if (recording.current === run) onError(error.message);
    } finally {
      if (recording.current === run) cancel();
    }
  }

  async function start() {
    if (recording.current || disabled) return;
    onError('');
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext || !window.AudioWorkletNode) {
      onError('Microphone recording is unavailable in this browser. Open Abyss on localhost in a browser with Web Audio support.');
      return;
    }
    const run = { request: new AbortController() };
    recording.current = run;
    setPhase('permission');
    setElapsed(0);
    onActiveChange(true);
    try {
      run.context = new AudioContext();
      const resumed = run.context.resume();
      resumed.catch(() => {});
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (recording.current !== run) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      run.stream = stream;
      await resumed;
      await run.context.audioWorklet.addModule('/dictation-processor.js');
      if (recording.current !== run) return;
      run.node = new AudioWorkletNode(run.context, 'abyss-dictation', {
        channelCount: 1, channelCountMode: 'explicit',
      });
      run.node.port.onmessage = event => finish(run, event.data);
      run.node.onprocessorerror = () => {
        if (recording.current !== run) return;
        onError('Microphone recording failed. Try again.');
        cancel();
      };
      for (const track of stream.getAudioTracks()) {
        track.onended = () => {
          if (recording.current !== run) return;
          onError('The microphone disconnected. Reconnect it and try again.');
          cancel();
        };
      }
      run.context.createMediaStreamSource(stream).connect(run.node);
      // The processor emits silence. A connected graph keeps capture running.
      run.node.connect(run.context.destination);
      run.started = performance.now();
      setPhase('recording');
    } catch (error) {
      if (recording.current !== run) return;
      onError(microphoneError(error));
      cancel();
    }
  }

  function stop() {
    const run = recording.current;
    if (phase !== 'recording' || !run?.node) return;
    setPhase('processing');
    run.stream.getTracks().forEach(track => track.stop());
    run.node.port.postMessage('finish');
    run.stopTimeout = setTimeout(() => {
      if (recording.current !== run) return;
      onError('The recording could not be finished. Try again.');
      cancel();
    }, 3000);
  }

  if (phase === 'idle') {
    return <button type="button" className="icon-button composer-icon" disabled={disabled}
      aria-label="Start dictation" title="Start dictation" onClick={start}><Icon name="microphone" size={19}/></button>;
  }
  const clock = `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, '0')}`;
  return <div className="dictation-controls">
    <span className={phase === 'recording' ? 'dictation-recording' : ''} role="status">
      {phase === 'permission' ? 'Waiting for microphone permission…' : phase === 'processing' ? 'Transcribing locally…' : 'Recording'}
    </span>
    {phase === 'recording' && <>
      <time aria-label="Recording time">{clock}</time>
      <button type="button" onClick={stop}>Stop</button>
    </>}
    <button type="button" onClick={cancel}>Cancel</button>
  </div>;
}
