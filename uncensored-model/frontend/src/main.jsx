import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';
import './models.css';
import Comparison from './Comparison';

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, { cache: 'no-store', ...options, headers: { 'Content-Type': 'application/json' } });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed. Check your input and try again.');
  }
  return response.status === 204 ? null : response.json();
}

function App() {
  const [conversations, setConversations] = useState([]);
  const [current, setCurrent] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [status, setStatus] = useState(null);
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState('');
  const [comparing, setComparing] = useState(false);
  const selectedModel = models.find(m => m.id === modelId);
  const modelRefresh = useRef(0);

  async function refreshModels() {
    const request = ++modelRefresh.current;
    const catalog = await api('/models');
    if (request !== modelRefresh.current) return;
    setModels(catalog.models);
    setModelId(old => {
      let saved = '';
      try { saved = localStorage.getItem('abyss-model') || localStorage.getItem('local-chat-model') || ''; } catch {}
      if (catalog.models.some(m => m.id === (old || saved))) return old || saved;
      const installed = catalog.models.filter(m => m.installed);
      return (installed.find(m => m.id === catalog.default_model) || installed[0])?.id || catalog.default_model;
    });
  }
  useEffect(() => {
    if (!modelId) return;
    try { localStorage.setItem('abyss-model', modelId); localStorage.removeItem('local-chat-model'); } catch {}
    let active = true;
    api(`/status?model_id=${encodeURIComponent(modelId)}`).then(value => { if (active) setStatus(value); }).catch(() => { if (active) setStatus({ ready: false, detail: 'Backend unavailable' }); });
    return () => { active = false; };
  }, [modelId, models]);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState('');
  const bottom = useRef(null);
  const selection = useRef(0);
  const active = conversations.find(c => c.id === current);
  const interrupted = messages.length > 0 && messages.at(-1).status !== 'complete';
  const bytes = new TextEncoder().encode(input.trim()).length;

  async function refresh() { setConversations(await api('/conversations')); }
  async function open(id) {
    const ticket = ++selection.current;
    setLoading(true); setError(''); setEditing(false); setInput('');
    try {
      const data = await api(`/conversations/${id}`);
      if (ticket === selection.current) { setCurrent(id); setMessages(data.messages); }
    } catch (e) { setError(e.message); }
    finally { if (ticket === selection.current) setLoading(false); }
  }
  useEffect(() => {
    (async () => {
      try {
        const list = await api('/conversations'); setConversations(list);
        if (list.length) await open(list[0].id);
      } catch (e) { setError(e.message); }
      finally { setLoading(false); }
    })();
    refreshModels().catch(e => setError(e.message));
    // Model installs happen outside this app. Re-check on return and
    // periodically while visible so an absent model never stays absent forever.
    const refreshInstalled = () => {
      if (document.visibilityState === 'visible') refreshModels().catch(() => {});
    };
    const timer = window.setInterval(refreshInstalled, 15000);
    window.addEventListener('focus', refreshInstalled);
    document.addEventListener('visibilitychange', refreshInstalled);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener('focus', refreshInstalled);
      document.removeEventListener('visibilitychange', refreshInstalled);
      modelRefresh.current++;
    };
  }, []);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: busy ? 'instant' : 'smooth' }); }, [messages]);

  async function newChat() {
    setLoading(true); setError('');
    try { const chat = await api('/conversations', { method: 'POST' }); await refresh(); await open(chat.id); }
    catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }
  async function rename(e) {
    e.preventDefault();
    try { await api(`/conversations/${current}`, { method: 'PATCH', body: JSON.stringify({ title }) }); await refresh(); setEditing(false); }
    catch (e) { setError(e.message); }
  }
  async function remove() {
    if (!window.confirm(`Delete “${active.title}” and its messages? This cannot be undone.`)) return;
    setLoading(true);
    try {
      await api(`/conversations/${current}`, { method: 'DELETE' });
      setCurrent(null); setMessages([]); setInput(''); setEditing(false); await refresh();
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }
  async function send(retry = false) {
    if (busy || loading || !current || !selectedModel?.installed || (!retry && (!bytes || bytes > 2500))) return;
    setBusy(true); setError('');
    const cid = current;
    const content = input.trim();
    try {
      const response = await fetch(`/api/conversations/${cid}/${retry ? 'retry' : 'messages'}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(retry ? { model_id: modelId } : { content, model_id: modelId }),
      });
      if (!response.ok) { const data = await response.json(); throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not send message.'); }
      if (!retry) setInput('');
      setMessages(old => [...(retry ? old.slice(0, -1) : [...old, { id: 'user-pending', role: 'user', content, status: 'complete' }]), { id: 'pending', role: 'assistant', content: '', status: 'streaming' }]);
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let buffer = '', completed = false;
      function consume(line) {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === 'delta') setMessages(old => old.map((m, i) => i === old.length - 1 ? { ...m, content: m.content + event.content } : m));
        if (event.type === 'done') completed = true;
        if (event.type === 'error') throw new Error(event.detail);
      }
      try {
        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value, { stream: !done });
          const lines = buffer.split('\n'); buffer = lines.pop(); lines.forEach(consume);
          if (done) { if (buffer) consume(buffer); break; }
        }
        if (!completed) throw new Error('Response interrupted. Retry to generate it again.');
      } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
    } catch (e) { setError(e.message); }
    finally {
      try { const chat = await api(`/conversations/${cid}`); setMessages(chat.messages); await refresh(); }
      catch { setError('Connection lost. Reload to restore saved messages.'); }
      api(`/status?model_id=${encodeURIComponent(modelId)}`).then(setStatus).catch(() => {});
      setBusy(false);
    }
  }

  return <div className="app">
    <aside>
      <div className="brand"><img className="brand-mascot" src="/abyss-icon.svg" width="30" height="30" alt=""/> Abyss <span className="private">PRIVATE</span></div>
      <button className="new-chat" onClick={newChat} disabled={busy || loading}><span>＋</span> New conversation</button>
      <div className="section-label">CONVERSATIONS <span>{conversations.length}</span></div>
      <nav aria-label="Conversations">{conversations.map(c => <button key={c.id} title={c.title} className={`conversation ${current === c.id ? 'selected' : ''}`} disabled={busy || loading} onClick={() => open(c.id)}><span className="chat-icon">▤</span><span>{c.title}</span></button>)}{!conversations.length && <p className="sidebar-empty">Your conversations will appear here.</p>}</nav>
      <div className="sidebar-footer"><span className="local-dot"/> Stored on this device <small>No accounts. No cloud.</small></div>
    </aside>
    <main>
      <header><div><strong>{active?.title || 'Abyss'}</strong><div className="model-controls"><select aria-label="Active text model" value={modelId} disabled={busy || loading} onChange={e => setModelId(e.target.value)}>{models.map(m => <option key={m.id} value={m.id}>{m.displayName}{m.installed ? '' : ' (not installed)'}</option>)}</select><button disabled={busy} onClick={() => setComparing(!comparing)}>{comparing ? 'Back to chat' : 'Compare models'}</button><button disabled={busy} onClick={() => refreshModels().catch(e => setError(e.message))}>Refresh Models</button></div></div>{current && <div className="actions"><button disabled={busy || loading} onClick={() => { setTitle(active.title); setEditing(!editing); }}>Rename</button><button disabled={busy || loading} onClick={remove}>Delete</button></div>}</header>
      {!comparing && editing && <form className="rename" onSubmit={rename}><input aria-label="Conversation title" autoFocus maxLength={100} value={title} onChange={e => setTitle(e.target.value)}/><button disabled={!title.trim()}>Save</button><button type="button" onClick={() => setEditing(false)}>Cancel</button></form>}
      <div className="chat-scroll" hidden={!comparing}><Comparison models={models} busy={busy} setBusy={setBusy}/></div>
      <div className="chat-scroll" hidden={comparing}>
        {!messages.length ? <section className="welcome"><div className="welcome-icon">◈</div><div className="eyebrow">A LITTLE SPACE TO THINK</div><h1>What’s on your mind?</h1><p>A conversation that stays on your computer.<br/>Ask a question, work through an idea, or start writing.</p>{!current && <button className="primary" disabled={loading} onClick={newChat}>Start a conversation <span>＋</span></button>}<div className="privacy-note"><span className="local-dot"/> Local model <span>·</span> Local history <span>·</span> Just you</div></section> : <div className="messages">{messages.map(m => <article key={m.id} className={`message ${m.role}`}><div className={`avatar ${m.role}`}>{m.role === 'user' ? 'Y' : '◈'}</div><div className="message-body"><div className="speaker">{m.role === 'user' ? 'You' : 'Assistant'}{m.status === 'streaming' && <span className="generating">Generating…</span>}</div><div className="message-text">{m.content || (m.status === 'streaming' ? 'Thinking…' : 'No response generated.')}</div>{['error', 'interrupted'].includes(m.status) && <small className="incomplete">Incomplete response</small>}</div></article>)}<div ref={bottom}/></div>}
      </div>
      <div className="composer-area" hidden={comparing}>
        {error && <div role="alert" className="error">{error}</div>}
        {interrupted && !busy && <div className="retry-row"><span>The last response did not finish.</span><button onClick={() => send(true)} disabled={loading || !selectedModel?.installed}>Retry response</button></div>}
        <form className="composer" onSubmit={e => { e.preventDefault(); send(); }}><textarea aria-label="Message" placeholder={current ? `Message ${selectedModel?.displayName || 'model'}...` : 'Create a conversation to begin'} value={input} disabled={!current || busy || loading || interrupted} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); } }}/><div className="composer-bottom"><span>Text only <span className="separator">/</span> {bytes > 2500 ? <span className="limit">Message too long ({bytes}/2500 bytes)</span> : 'Shift + Enter for a new line'}</span><button className="send" aria-label="Send message" disabled={!selectedModel?.installed || !current || !bytes || bytes > 2500 || busy || loading || interrupted}>{busy ? '…' : '↑'}</button></div></form>
        <div className="bottom-note"><span className={`local-dot ${status?.ready ? '' : 'offline'}`}/>{status?.ready ? 'Running locally · Conversations saved automatically' : status?.detail || 'Checking local runtime…'}</div>
      </div>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
