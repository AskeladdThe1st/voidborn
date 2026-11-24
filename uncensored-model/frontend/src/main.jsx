import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';
import './models.css';
import Comparison from './Comparison';
import Icon from './Icon';
import MessageContent from './MessageContent';
import Dictation from './Dictation';
import { ACCEPT_ATTACHMENTS, MAX_ATTACHMENTS, DraftAttachments, MessageAttachments, attachmentError, isImageAttachment, pastedImages } from './Attachments';

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
  const [historyOpen, setHistoryOpen] = useState(false);
  const [runtimeAvailable, setRuntimeAvailable] = useState(null);
  const [vision, setVision] = useState(null);
  const [attachments, setAttachments] = useState([]);
  const [dictating, setDictating] = useState(false);
  const composerInput = useRef(null);
  const generation = useRef(null);
  const [stopping, setStopping] = useState(false);
  const filePicker = useRef(null);
  const historyToggle = useRef(null);
  const historyClose = useRef(null);
  const selectedModel = models.find(m => m.id === modelId);
  const modelRefresh = useRef(0);

  function closeHistory() {
    setHistoryOpen(false);
    historyToggle.current?.focus();
  }
  useEffect(() => {
    if (historyOpen) historyClose.current?.focus();
  }, [historyOpen]);

  async function refreshModels() {
    const request = ++modelRefresh.current;
    const catalog = await api('/models');
    if (request !== modelRefresh.current) return;
    setModels(catalog.models);
    setRuntimeAvailable(catalog.available);
    setVision(catalog.vision);
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
  const runtimeLabel = runtimeAvailable === null ? 'Checking connection…' : runtimeAvailable
    ? (vision?.installed || models.some(m => m.installed) ? 'Connected' : 'Connected · no models installed') : 'Offline';
  const latest = messages.at(-1);
  const interrupted = Boolean(latest && (latest.status !== 'complete' ||
    (latest.role === 'assistant' && !latest.content.trim())));
  const bytes = new TextEncoder().encode(input.trim()).length;
  const hasImage = attachments.some(item => isImageAttachment(item.file));
  const retryHasImage = interrupted && messages.findLast(m => m.role === 'user')?.attachments?.some(isImageAttachment);
  const canEdit = Boolean(current && !busy && !loading && !interrupted);
  const canSend = canEdit && !dictating && (hasImage ? vision?.installed : selectedModel?.installed) &&
    (bytes > 0 || attachments.length > 0) && bytes <= 2500;
  const canRetry = !busy && !loading && (retryHasImage ? vision?.installed : selectedModel?.installed);

  useLayoutEffect(() => {
    const textarea = composerInput.current;
    if (!textarea || comparing) return;
    const resize = () => {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight}px`;
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [input, comparing]);

  function addAttachments(files) {
    if (!canEdit) return;
    const accepted = [], errors = [];
    for (const file of files) {
      const invalid = attachmentError(file);
      if (invalid) errors.push(invalid);
      else if (attachments.length + accepted.length >= MAX_ATTACHMENTS) {
        errors.push('You can attach up to five files per message.');
        break;
      } else {
        accepted.push({ id: crypto.randomUUID(), file });
      }
    }
    setAttachments([...attachments, ...accepted]);
    setError(errors.join(' '));
  }

  async function refresh() { setConversations(await api('/conversations')); }
  async function open(id) {
    const ticket = ++selection.current;
    setLoading(true); setError(''); setEditing(false); setInput('');
    try {
      const data = await api(`/conversations/${id}`);
      if (ticket === selection.current) {
        setCurrent(id); setMessages(data.messages); setComparing(false); setAttachments([]);
        if (historyOpen) closeHistory();
      }
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
    if (!window.confirm(`Delete “${active.title}”, its messages, and attachments? This cannot be undone.`)) return;
    setLoading(true);
    try {
      await api(`/conversations/${current}`, { method: 'DELETE' });
      setCurrent(null); setMessages([]); setInput(''); setAttachments([]); setEditing(false); await refresh();
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }
  async function send(retry = false) {
    if (generation.current || (retry ? !canRetry : !canSend)) return;
    const controller = new AbortController();
    generation.current = controller;
    setBusy(true); setError('');
    const cid = current;
    const content = input.trim();
    try {
      const multipart = !retry && attachments.length > 0;
      let body, headers;
      if (multipart) {
        body = new FormData();
        body.append('content', content);
        if (modelId) body.append('model_id', modelId);
        attachments.forEach(({ file }) => body.append('files', file));
      } else {
        headers = { 'Content-Type': 'application/json' };
        body = JSON.stringify(retry ? { model_id: modelId } : { content, model_id: modelId });
      }
      const path = retry ? 'retry' : multipart ? 'messages/attachments' : 'messages';
      const response = await fetch(`/api/conversations/${cid}/${path}`, {
        method: 'POST', headers, body, signal: controller.signal,
      });
      if (!response.ok) { const data = await response.json(); throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not send message.'); }
      if (!retry) { setInput(''); setAttachments([]); }
      setMessages(old => [...(retry ? old.slice(0, -1) : [...old, { id: 'user-pending', role: 'user', content, status: 'complete' }]), { id: 'pending', role: 'assistant', content: '', status: 'streaming' }]);
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let buffer = '', completed = false;
      function consume(line) {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === 'start' && event.attachments) {
          setMessages(old => old.map(m => m.id === 'user-pending' ? { ...m, attachments: event.attachments } : m));
        }
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
    } catch (e) {
      if (!controller.signal.aborted) setError(e.message);
    }
    finally {
      try {
        let chat = await api(`/conversations/${cid}`);
        // Let the backend finish saving the interrupted reply after disconnect.
        for (let attempt = 0; controller.signal.aborted && chat.messages.at(-1)?.status === 'streaming' && attempt < 10; attempt++) {
          await new Promise(resolve => setTimeout(resolve, 75));
          chat = await api(`/conversations/${cid}`);
        }
        setMessages(chat.messages);
        await refresh();
      }
      catch { setError(currentError => currentError || 'Connection lost. Reload to restore saved messages.'); }
      api(`/status?model_id=${encodeURIComponent(modelId)}`).then(setStatus).catch(() => {});
      generation.current = null;
      setStopping(false);
      setBusy(false);
    }
  }

  return <div className="app">
    {historyOpen && <button type="button" className="history-backdrop" tabIndex={-1} aria-label="Dismiss conversation history" onClick={closeHistory}/>}
    <aside id="history" className={historyOpen ? 'history-open' : ''} onKeyDown={e => { if (historyOpen && e.key === 'Escape') closeHistory(); }}>
      <div className="brand">
        <img className="insignia" src="/art/insignia.svg" width="90" height="112" alt=""/>
        <div><img className="wordmark" src="/art/wordmark.svg" width="158" height="28" alt="Abyss"/><div className="brand-caption">LOCAL INTELLIGENCE SYSTEM</div></div>
        <button ref={historyClose} className="icon-button history-close" aria-label="Close conversation history" onClick={closeHistory}><Icon name="close"/></button>
      </div>
      <button className="new-chat" onClick={newChat} disabled={busy || loading}><Icon name="plus"/> New Chat</button>
      <div className="section-label">Recent conversations</div>
      <nav aria-label="Conversations">
        {conversations.map(c => <button key={c.id} title={c.title} aria-current={current === c.id ? 'true' : undefined}
          className={`conversation ${current === c.id ? 'selected' : ''}`} disabled={busy || loading} onClick={() => open(c.id)}>
          <Icon name="folder" size={19}/><span>{c.title}</span>
        </button>)}
        {!conversations.length && <p className="sidebar-empty">No conversations yet.</p>}
      </nav>
      <div className="sidebar-footer">
        <Icon name="cube" size={28}/>
        <div><strong>Ollama (Local)</strong><small><span className={`local-dot ${runtimeAvailable ? '' : 'offline'}`}/>{runtimeLabel}</small></div>
      </div>
    </aside>

    <main>
      <header className="topbar">
        <button ref={historyToggle} className="icon-button history-toggle" aria-label="Open conversation history" aria-expanded={historyOpen} aria-controls="history" onClick={() => historyOpen ? closeHistory() : setHistoryOpen(true)}><Icon name="menu"/></button>
        <span className="mobile-brand">ABYSS</span>
        <fieldset className="model-picker">
          <legend>Model</legend>
          <Icon name="cube"/>
          <select aria-label="Active text model" value={modelId} disabled={busy || loading || !models.length} onChange={e => setModelId(e.target.value)}>
            {!models.length && <option value="">Loading models…</option>}
            {models.map(m => <option key={m.id} value={m.id}>{m.displayName}{m.installed ? '' : ' (not installed)'}</option>)}
          </select>
        </fieldset>
        <div className="actions">
          <button className="icon-button" title={comparing ? 'Back to chat' : 'Compare models'} aria-label={comparing ? 'Back to chat' : 'Compare models'} aria-pressed={comparing} disabled={busy} onClick={() => setComparing(!comparing)}><Icon name="compare"/></button>
          <button className="icon-button" title="Refresh models" aria-label="Refresh models" disabled={busy} onClick={() => refreshModels().catch(e => setError(e.message))}><Icon name="refresh"/></button>
          {current && !comparing && <>
            <button className="icon-button" title="Rename conversation" aria-label="Rename conversation" disabled={busy || loading} onClick={() => { setTitle(active.title); setEditing(!editing); }}><Icon name="edit" size={20}/></button>
            <button className="icon-button" title="Delete conversation" aria-label="Delete conversation" disabled={busy || loading} onClick={remove}><Icon name="trash" size={20}/></button>
          </>}
        </div>
      </header>

      <div className={comparing ? 'channel' : 'channel conversation-channel'}>
        <h1><span aria-hidden="true">//</span> {comparing ? 'Model comparison' : 'Conversation'}</h1>
        {!comparing && active && <span className="channel-title" title={active.title}>{active.title}</span>}
      </div>
      {!comparing && editing && <form className="rename" onSubmit={rename}>
        <input aria-label="Conversation title" autoFocus maxLength={100} value={title} onChange={e => setTitle(e.target.value)}/>
        <button disabled={!title.trim()}>Save</button><button type="button" onClick={() => setEditing(false)}>Cancel</button>
      </form>}
      <div className="chat-scroll" hidden={!comparing}>
        {error && <div role="alert" className="error comparison-error">{error}</div>}
        <Comparison models={models} busy={busy} setBusy={setBusy}/>
      </div>
      <div className="chat-scroll" hidden={comparing}>
        {!messages.length ? <section className="welcome">
          <img className="pixel-portrait" src="/art/stormtrooper-reference.png" alt="" width="72" height="72"/>
          <h2>{loading ? 'Loading conversation…' : 'Abyss is listening'}</h2>
          {!current && !loading && <button className="primary" onClick={newChat}>Start a conversation <Icon name="arrow" size={16}/></button>}
        </section> : <div className="messages">
          {messages.map(m => <article key={m.id} className={`message ${m.role}`}>
            <div className="avatar"><img className="pixel-portrait" src={m.role === 'user' ? '/art/vader-reference.png' : '/art/stormtrooper-reference.png'} width="48" height="48" alt=""/></div>
            <div className="message-body">
              <div className="speaker">
                <strong>{m.role === 'user' ? 'YOU' : 'ABYSS'}</strong>
                <span className="speaker-role">/ {m.role === 'user' ? 'COMMANDER' : 'THE ENTITY'}</span>
                {m.status === 'streaming' && <span className="generating">Generating…</span>}
                {m.created_at && <time dateTime={m.created_at}>{new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time>}
              </div>
              {(m.content.trim() || m.status === 'streaming') && <MessageContent>{m.content || 'Thinking…'}</MessageContent>}
              {m.attachments?.length > 0 && <MessageAttachments conversationId={current} files={m.attachments}/>}
              {m.role === 'assistant' && m.status !== 'streaming' && (m.status !== 'complete' || !m.content.trim()) &&
                <small className="incomplete">{m.content.trim() ? 'Incomplete response' : 'The model did not return an answer.'}</small>}
            </div>
          </article>)}
          <div ref={bottom}/>
        </div>}
      </div>
      <div className="composer-area" hidden={comparing}
        onDragOver={e => {
          if (e.dataTransfer.types.includes('Files')) {
            e.preventDefault();
            e.dataTransfer.dropEffect = canEdit ? 'copy' : 'none';
          }
        }}
        onDrop={e => {
          if (e.dataTransfer.files.length) { e.preventDefault(); addAttachments(e.dataTransfer.files); }
        }}>
        {error && <div role="alert" className="error">{error}</div>}
        {interrupted && !busy && <div className="retry-row"><span>The last response did not finish.</span><button onClick={() => send(true)} disabled={!canRetry}>Retry response</button></div>}
        {attachments.length > 0 && <DraftAttachments items={attachments} disabled={busy} onRemove={id => {
          setAttachments(items => items.filter(item => item.id !== id)); setError('');
        }}/>}
        <form className="composer" onSubmit={e => { e.preventDefault(); send(); }}>
          <input type="file" ref={filePicker} accept={ACCEPT_ATTACHMENTS} multiple hidden
            onChange={e => { addAttachments(e.target.files); e.target.value = ''; }}/>
          <textarea ref={composerInput} rows={1} aria-label="Message" aria-describedby="composer-status" placeholder="Message Abyss..." value={input}
            disabled={!canEdit} onChange={e => setInput(e.target.value)}
            onPaste={e => {
              const images = pastedImages(e.clipboardData);
              if (images.length) { e.preventDefault(); addAttachments(images); }
            }}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); } }}/>
          <div className="composer-tools">
            <button type="button" className="icon-button composer-icon" aria-label="Attach files" title="Attach files"
              disabled={!canEdit} onClick={() => filePicker.current?.click()}><Icon name="plus" size={20}/></button>
            <div className="composer-actions">
              {!comparing && <Dictation key={current} disabled={!canEdit} onActiveChange={setDictating} onError={setError}
                onTranscript={text => {
                  setInput(previous => previous && !/\s$/.test(previous) ? `${previous} ${text}` : previous + text);
                  composerInput.current?.focus();
                }}/>}
              {busy ? <button type="button" className="send" title="Stop generation" aria-label="Stop generation"
                disabled={stopping || !generation.current} onClick={() => {
                  setStopping(true);
                  generation.current?.abort();
                }}><Icon name="stop" size={16}/></button>
                : <button type="submit" className="send" title="Send message" aria-label="Send message"
                  disabled={!canSend}><Icon name="send" size={20}/></button>}
            </div>
          </div>
        </form>
        <div className="composer-status" id="composer-status">
          <span>{hasImage || retryHasImage
            ? (vision?.installed ? 'Images use qwen3-vl:4b locally' : 'Images require qwen3-vl:4b in local Ollama. Install it, then refresh models.')
            : (status?.ready ? 'Running locally · Conversations saved automatically' : status?.detail || 'Checking local runtime…')}</span>
          <span className={bytes > 2500 ? 'limit' : 'keyboard-hint'}>{bytes > 2500 ? `Message too long (${bytes}/2500 bytes)` : 'Shift + Enter for a new line'}</span>
        </div>
      </div>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
