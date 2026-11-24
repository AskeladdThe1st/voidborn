import React from 'react';

const paths = {
  plus: 'M12 4v16M4 12h16',
  folder: 'M3 6h6l2 2h10v11H3zM9 14h6',
  cube: 'm12 2 9 5v10l-9 5-9-5V7zM3 7l9 5 9-5m-9 5v10',
  compare: 'M3 5h18v14H3zM12 5v14M6 10h3m6 4h3',
  refresh: 'M20 10a8 8 0 0 0-14-4L3 9m0-5v5h5m-4 5a8 8 0 0 0 14 4l3-3m0 5v-5h-5',
  edit: 'm4 16-1 5 5-1L21 7l-4-4zM14 6l4 4',
  trash: 'M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7',
  paperclip: 'm20 11-8 8a5 5 0 0 1-7-7l9-9a3.5 3.5 0 0 1 5 5l-9 9a2 2 0 0 1-3-3l8-8',
  file: 'M5 3h9l5 5v13H5zM14 3v6h5M8 13h8m-8 4h8',
  image: 'M3 4h18v16H3zM3 16l5-5 4 4 3-3 6 6M16 8h.01',
  send: 'm3 3 18 9-18 9 4-9zM7 12h14',
  menu: 'M4 6h16M4 12h16M4 18h16',
  close: 'm5 5 14 14M19 5 5 19',
  arrow: 'm9 5 7 7-7 7',
};

export default function Icon({ name, size = 22 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]}/></svg>;
}
