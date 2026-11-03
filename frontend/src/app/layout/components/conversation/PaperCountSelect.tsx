import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { BookOpen, Check, ChevronDown } from 'lucide-react';

export function PaperCountSelect({ value, disabled, onChange }: {
  value: number; disabled: boolean; onChange: (value: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ left: 0, bottom: 0 });
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const listId = useId();
  const values = [3, 5, 8].includes(value) ? [3, 5, 8] : [value, 3, 5, 8];

  const close = (restoreFocus = false) => {
    setOpen(false);
    if (restoreFocus) trigger.current?.focus();
  };
  useLayoutEffect(() => {
    if (!open || !trigger.current) return;
    const bounds = trigger.current.getBoundingClientRect();
    setPosition({ left: Math.max(8, Math.min(bounds.left, window.innerWidth - 196)), bottom: window.innerHeight - bounds.top + 8 });
    menu.current?.querySelector<HTMLElement>('[aria-selected="true"]')?.focus();
  }, [open, value]);
  useEffect(() => { if (disabled) setOpen(false); }, [disabled]);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!menu.current?.contains(event.target as Node) && !trigger.current?.contains(event.target as Node)) setOpen(false);
    };
    const reposition = () => setOpen(false);
    document.addEventListener('pointerdown', outside);
    window.addEventListener('resize', reposition);
    return () => { document.removeEventListener('pointerdown', outside); window.removeEventListener('resize', reposition); };
  }, [open]);

  return <>
    <button ref={trigger} type="button" className={`paper-count-trigger ${open ? 'is-open' : ''}`}
      aria-label="精选论文数量" aria-haspopup="listbox" aria-expanded={open} aria-controls={open ? listId : undefined}
      disabled={disabled} onClick={() => setOpen(current => !current)}
      onKeyDown={event => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); setOpen(true); }
      }}>
      <BookOpen size={14}/><span>精选 <strong>{value}</strong> 篇</span><ChevronDown size={13}/>
    </button>
    {open && createPortal(<div ref={menu} id={listId} role="listbox" aria-label="精选论文数量选项"
      className="paper-count-menu" style={position}
      onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false); }}
      onKeyDown={event => {
        const options = Array.from(menu.current?.querySelectorAll<HTMLButtonElement>('[role="option"]') ?? []);
        const current = options.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === 'ArrowDown' ? (current + 1) % options.length
          : event.key === 'ArrowUp' ? (current - 1 + options.length) % options.length
            : event.key === 'Home' ? 0 : event.key === 'End' ? options.length - 1 : -1;
        if (next >= 0) { event.preventDefault(); options[next]?.focus(); }
        if (event.key === 'Escape') { event.preventDefault(); close(true); }
        if (event.key === 'Tab') close(true);
      }}>
      <div className="paper-count-menu-label">精选论文数量</div>
      {values.map(count => <button type="button" role="option" key={count} aria-selected={value === count}
        tabIndex={-1} onClick={() => { onChange(count); close(true); }}>
        <span><strong>{count}</strong> 篇论文</span>{value === count && <Check size={15}/>}
      </button>)}
    </div>, document.body)}
  </>;
}
