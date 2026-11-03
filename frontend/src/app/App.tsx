import { useLayoutEffect } from 'react';
import { WorkspaceShell } from './layout/components/WorkspaceShell';
import { useLayoutState } from './layout/stores/layoutStore';

export default function App() {
  const { theme } = useLayoutState();
  useLayoutEffect(() => {
    const system = window.matchMedia('(prefers-color-scheme: dark)');
    const apply = () => {
      const resolved = theme === 'system' ? (system.matches ? 'dark' : 'light') : theme;
      document.documentElement.dataset.theme = resolved;
      document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')
        ?.setAttribute('content', resolved === 'dark' ? '#161513' : '#f5f3ee');
    };
    apply();
    system.addEventListener('change', apply);
    return () => system.removeEventListener('change', apply);
  }, [theme]);
  return <WorkspaceShell/>;
}
