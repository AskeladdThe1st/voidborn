import React from 'react';
import Markdown from 'react-markdown';

const components = {
  // Remote Markdown images must never cause the browser to make a request.
  img: ({ alt }) => alt ? <span>{alt}</span> : null,
  a: ({ href, children }) => /^https?:\/\//i.test(href || '')
    ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
    : <span>{children}</span>,
};

export default function MessageContent({ children }) {
  return <div className="message-text"><Markdown skipHtml components={components}>{children}</Markdown></div>;
}
