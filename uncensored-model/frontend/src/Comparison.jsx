import React, { useState } from 'react';
import MessageContent from './MessageContent';
import { MAX_PROMPT_BYTES } from './limits';

const categories = [
  ['acceptance', 'Acceptance', '1 = unnecessary refusal; 10 = directly addresses the request'],
  ['understanding', 'Prompt understanding', '1 = misunderstood; 10 = followed the request closely'],
  ['reasoning', 'Reasoning quality', '1 = incoherent or flawed; 10 = sound and well reasoned'],
  ['reliability', 'Hallucination / factual reliability', '1 = invented or unreliable claims; 10 = accurate and honest about uncertainty'],
  ['speed', 'Response speed', '1 = too slow; 10 = very responsive'],
];

export default function Comparison({ models, busy, setBusy }) {
  const [prompt, setPrompt] = useState('');
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [testedPrompt, setTestedPrompt] = useState('');
  const bytes = new TextEncoder().encode(prompt).length;
  const installed = models.filter(m => m.installed);

  function update(id, fields) {
    setResults(old => old.map(r => r.id === id ? { ...r, ...fields } : r));
  }

  async function run(event) {
    event.preventDefault();
    if (busy || !prompt.trim() || bytes > MAX_PROMPT_BYTES || !installed.length) return;
    setBusy(true); setError(''); setTestedPrompt(prompt);
    setResults(installed.map(m => ({ id: m.id, label: m.displayName, content: '', state: 'Waiting', ratings: {} })));
    try {
      const response = await fetch('/api/compare', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: prompt }),
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(typeof data.detail === 'string' ? data.detail : 'Comparison failed.');
      }
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let buffer = '', complete = false;
      function consume(line) {
        if (!line.trim()) return;
        const item = JSON.parse(line);
        if (item.type === 'start') update(item.model_id, { state: 'Generating' });
        if (item.type === 'delta') setResults(old => old.map(r => r.id === item.model_id ? { ...r, content: r.content + item.content } : r));
        if (item.type === 'result') update(item.model_id, { state: 'Complete', seconds: item.seconds, first: item.first_token_seconds });
        if (item.type === 'model_error') update(item.model_id, { state: 'Failed', error: item.detail, seconds: item.seconds });
        if (item.type === 'done') complete = true;
      }
      try {
        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value, { stream: !done });
          const lines = buffer.split('\n'); buffer = lines.pop(); lines.forEach(consume);
          if (done) { if (buffer) consume(buffer); break; }
        }
        if (!complete) throw new Error('Comparison was interrupted. Run it again.');
      } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
    } catch (e) {
      setError(e.message);
      setResults(old => old.map(r => ['Waiting', 'Generating'].includes(r.state) ? { ...r, state: 'Interrupted' } : r));
    } finally { setBusy(false); }
  }

  return <section className="comparison" aria-label="Local model comparison">
    <h2>Compare local models</h2>
    <p>The same prompt, fresh context, one model at a time. Timing includes loading. Results and your ratings stay in this panel until you reload.</p>
    <div className="model-installations">{models.map(m => <div key={m.id}><strong>{m.displayName}</strong> — {m.installed ? 'Installed' : 'Not installed'}<small>Temperature {m.options.temperature} · top_p {m.options.top_p}</small></div>)}</div>
    <form onSubmit={run}>
      <textarea aria-label="Comparison prompt" placeholder="Enter a factual question or a prompt with a known answer…" value={prompt} disabled={busy} onChange={e => setPrompt(e.target.value)}/>
      <button className="primary" disabled={busy || !installed.length || !prompt.trim() || bytes > MAX_PROMPT_BYTES}>{busy ? 'Comparing…' : `Compare ${installed.length} installed model${installed.length === 1 ? '' : 's'}`}</button>
      {bytes > MAX_PROMPT_BYTES && <p className="limit">Limit: 128 KiB.</p>}
    </form>
    {error && <p className="error" role="alert">{error}</p>}
    {results.length > 0 && <><p className="tested-prompt"><strong>Test prompt:</strong> {testedPrompt}</p><p>Score each category from 1 (poor) to 10 (excellent). Higher factual reliability means fewer hallucinations. Leave a score unrated if you cannot judge it. Ratings are manual; Abyss does not rank models or select a winner.</p></>}
    <div className="comparison-results">{results.map(r => <article className="comparison-result" key={r.id}>
      <h3>{r.label}</h3><div className="result-timing">{r.state}{r.seconds != null && ` · Total ${r.seconds.toFixed(2)}s`}{r.first != null && ` · First token ${r.first.toFixed(2)}s`}</div>
      <MessageContent>{r.content}</MessageContent>{r.error && <p className="error">{r.error}</p>}
      {categories.map(([key, label, guide]) => <div className="rating" key={key}>
        <label>{label}<select aria-label={`${r.label}: ${label}`} value={r.ratings[key] || ''} disabled={busy || r.state !== 'Complete'} onChange={e => update(r.id, { ratings: { ...r.ratings, [key]: e.target.value } })}>
          <option value="">Unrated</option>{Array.from({ length: 10 }, (_, i) => <option key={i + 1} value={i + 1}>{i + 1}</option>)}
        </select></label><small>{guide}</small>
      </div>)}
    </article>)}</div>
  </section>;
}
