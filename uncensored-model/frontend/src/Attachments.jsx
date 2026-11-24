import React, { useEffect, useState } from 'react';
import Icon from './Icon';

const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'webp']);
const CLIPBOARD_IMAGES = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp' };
const DOCUMENT_EXTENSIONS = new Set(['pdf', 'docx', 'txt']);
const MIB = 1024 * 1024;
export const MAX_ATTACHMENTS = 5;
export const ACCEPT_ATTACHMENTS = '.png,.jpg,.jpeg,.webp,.pdf,.docx,.txt';

function extension(file) {
  return file.name.split('.').pop().toLowerCase();
}

export function isImageAttachment(file) {
  return file.mime ? file.mime.startsWith('image/') : IMAGE_EXTENSIONS.has(extension(file));
}

export function pastedImages(clipboard) {
  const files = Array.from(clipboard.files || []);
  if (!files.length) {
    for (const item of clipboard.items || []) {
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
  }
  return files.filter(file => file.type.startsWith('image/')).map(file => {
    const ext = CLIPBOARD_IMAGES[file.type];
    return ext && !isImageAttachment(file) ? new File([file], `Screenshot.${ext}`, { type: file.type }) : file;
  });
}

export function attachmentError(file) {
  const image = isImageAttachment(file);
  if (!image && !DOCUMENT_EXTENSIONS.has(extension(file))) return `${file.name}: Unsupported file type.`;
  if (!file.size) return `${file.name}: The file is empty.`;
  const max = image ? 10 : 20;
  if (file.size > max * MIB) return `${file.name}: Exceeds the ${max} MB limit.`;
  return '';
}

function sizeLabel(bytes) {
  return bytes >= MIB ? `${(bytes / MIB).toFixed(1)} MB` : `${Math.max(1, Math.ceil(bytes / 1024))} KB`;
}

function DraftAttachment({ item, onRemove, disabled }) {
  const [preview, setPreview] = useState('');
  const { file } = item;
  useEffect(() => {
    if (!isImageAttachment(file)) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return <li className="attachment">
    {preview ? <img className="attachment-preview" src={preview} alt=""/> : <Icon name="file" size={22}/>}
    <div className="attachment-info">
      <span title={file.name}>{file.name}</span><small>{sizeLabel(file.size)}</small>
    </div>
    <button type="button" className="icon-button attachment-remove" aria-label={`Remove ${file.name}`} onClick={onRemove} disabled={disabled}>
      <Icon name="close" size={17}/>
    </button>
  </li>;
}

export function DraftAttachments({ items, onRemove, disabled }) {
  return <ul className="attachment-list" aria-label="Selected attachments">
    {items.map(item => <DraftAttachment key={item.id} item={item} onRemove={() => onRemove(item.id)} disabled={disabled}/>)}
  </ul>;
}

export function MessageAttachments({ conversationId, files }) {
  return <ul className="attachment-list" aria-label="Message attachments">
    {files.map(file => {
      const url = `/api/conversations/${encodeURIComponent(conversationId)}/attachments/${encodeURIComponent(file.id)}`;
      const image = isImageAttachment(file);
      return <li className="attachment" key={file.id}>
        <Icon name={image ? 'image' : 'file'} size={22}/>
        <div className="attachment-info">
          <span title={file.filename}>{file.filename}</span>
          <small>{sizeLabel(file.size)}</small>
          {file.warning && <small className="limit">{file.warning}</small>}
        </div>
        {image && <a href={url} target="_blank" rel="noopener noreferrer" aria-label={`Open ${file.filename}`}>Open</a>}
        <a href={`${url}?download=true`} download aria-label={`Download ${file.filename}`}>Download</a>
      </li>;
    })}
  </ul>;
}
