/** Actual Electron/preload/Python/MCP fixture integration. Never uses live keys. */
import { _electron as electron, expect } from '@playwright/test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile, stat, utimes } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const packaged = process.argv[2];
const dev = !packaged && process.env.PAPER_AGENT_E2E_DEV === '1';
// Offline/restricted networks cannot reach the GitHub Electron release CDN, so
// prefer the binary already installed by the pinned `electron` devDependency.
function localElectronExecutable() {
  if (process.env.PAPER_AGENT_ELECTRON_EXE) return process.env.PAPER_AGENT_ELECTRON_EXE;
  const candidate = path.join(root, 'node_modules', 'electron', 'dist',
    process.platform === 'win32' ? 'electron.exe' : 'electron');
  return existsSync(candidate) ? candidate : undefined;
}
const devServer = dev ? await (await import('vite')).createServer({
  configFile: path.join(root, 'frontend/vite.config.ts'), root: path.join(root, 'frontend'),
  server: { host: '127.0.0.1', port: 5173, strictPort: true },
}) : null;
await devServer?.listen();
const output = path.join(root, 'data/validation/v020', `${packaged ? 'packaged' : dev ? 'dev' : 'source'}-${Date.now()}`);
const profile = path.join(output, 'profile');
await mkdir(profile, { recursive: true });
const env = { ...process.env, PAPER_AGENT_DESKTOP_DATA_DIR: profile,
  LLM_API_KEY: 'e2e-deepseek-placeholder', DEEPSEEK_API_KEY: '',
  OPENALEX_API_KEY: 'e2e-openalex-placeholder', PAPER_AGENT_MODE: 'demo',
  PAPER_AGENT_RENDERER_URL: dev ? 'http://127.0.0.1:5173' : '', PYTHONUTF8: '1' };
delete env.ELECTRON_RUN_AS_NODE;
const app = await electron.launch({
  ...(packaged ? { executablePath: path.resolve(packaged), args: [`--user-data-dir=${path.join(output, 'chromium')}`] }
    : { args: ['.', `--user-data-dir=${path.join(output, 'chromium')}`], executablePath: localElectronExecutable() }),
  cwd: root, env, timeout: 90_000, chromiumSandbox: true,
});
const report = { fixture: true, live_api_calls: 0, packaged: Boolean(packaged), checks: [] };
const errors = [];
let privateEndpoint;
let failure;
try {
  const page = await app.firstWindow({ timeout: 90_000 });
  page.on('pageerror', error => errors.push(error.message));
  await expect(page.getByRole('main', { name: '研究对话' })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole('heading', { name: '从一个研究问题开始' })).toBeVisible();
  const parallelSwitch = page.getByRole('switch', { name: '并行优化' });
  await expect(parallelSwitch).not.toBeChecked();
  await parallelSwitch.focus();
  await page.keyboard.press('Space');
  await expect(parallelSwitch).toBeChecked();
  await parallelSwitch.uncheck();
  report.checks.push('parallel-switch-visible-keyboard-operable-and-off-by-default');
  const paperCount = page.getByRole('button', { name: '精选论文数量', exact: true });
  await paperCount.focus();
  await page.keyboard.press('ArrowDown');
  await expect(page.getByRole('listbox', { name: '精选论文数量选项' })).toBeVisible();
  await page.keyboard.press('End');
  await page.keyboard.press('Enter');
  await expect(paperCount).toContainText('8');
  await expect(paperCount).toBeFocused();
  await paperCount.click();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('listbox')).toHaveCount(0);
  await paperCount.click();
  await page.getByRole('option', { name: '3 篇论文', exact: true }).click();
  await expect(paperCount).toContainText('3');
  report.checks.push('paper-count-popup-keyboard-selection-escape-and-focus-return');
  const info = await page.evaluate(() => window.paperAgent.info());
  assert.equal(info.api_version, 3);
  assert.equal(info.version, '1.0.0');
  assert.equal(path.resolve(info.data_root), profile);
  assert.equal('token' in info, false);
  const isolation = await page.evaluate(() => ({ node: typeof window.require, process: typeof window.process,
    bridge: Boolean(window.paperAgent), transport: 'endpoint' in window.paperAgent }));
  assert.deepEqual(isolation, { node: 'undefined', process: 'undefined', bridge: true, transport: false });
  const prefs = await app.evaluate(({ BrowserWindow }) => {
    const p = BrowserWindow.getAllWindows()[0].webContents.getLastWebPreferences();
    return { sandbox: p.sandbox, contextIsolation: p.contextIsolation, nodeIntegration: p.nodeIntegration };
  });
  assert.deepEqual(prefs, { sandbox: true, contextIsolation: true, nodeIntegration: false });
  if (packaged) assert.equal(new URL(page.url()).protocol, 'file:');
  const nativeMenu = await app.evaluate(({ Menu, BrowserWindow }) => ({
    menu: Menu.getApplicationMenu(), visible: BrowserWindow.getAllWindows()[0].isMenuBarVisible(),
  }));
  assert.equal(nativeMenu.menu, null);
  assert.equal(nativeMenu.visible, false);
  assert.equal(await page.locator('.dock-tab').count(), 1);
  assert.match(await page.locator('.dock-tab').first().innerText(), /论文/);
  report.checks.push('native-menu-removed-default-papers-only');

  // Sidebar regression: stable toggle, visible rail control, responsive main area.
  const sidebar = page.locator('.sidebar');
  const mainArea = page.locator('.conversation-shell');
  const expandedMain = await mainArea.boundingBox();
  await page.getByRole('button', { name: '收起侧栏', exact: true }).click();
  await expect(sidebar).toHaveClass(/sidebar-compact/);
  await expect.poll(async () => (await sidebar.boundingBox())?.width).toBe(58);
  const collapsedMain = await mainArea.boundingBox();
  assert.ok(collapsedMain.width > expandedMain.width);
  const expand = page.getByRole('button', { name: '展开侧栏', exact: true });
  await expect(expand).toBeVisible();
  const [sideBox, expandBox] = await Promise.all([sidebar.boundingBox(), expand.boundingBox()]);
  assert.ok(expandBox.x >= sideBox.x && expandBox.x + expandBox.width <= sideBox.x + sideBox.width);
  await expand.click();
  await expect(sidebar).not.toHaveClass(/sidebar-compact/);
  for (let index = 0; index < 9; index += 1) {
    const railExpand = page.getByRole('button', { name: '展开侧栏', exact: true });
    if (await railExpand.isVisible()) await railExpand.click();
    else await page.getByRole('button', { name: '收起侧栏', exact: true }).click();
  }
  await expect(sidebar).toHaveClass(/sidebar-compact/);
  await page.getByRole('button', { name: '展开侧栏', exact: true }).click();
  await expect(sidebar).not.toHaveClass(/sidebar-compact/);
  await page.getByRole('button', { name: '收起侧栏', exact: true }).click();
  await page.waitForTimeout(260);
  await page.reload();
  await expect(page.getByRole('main', { name: '研究对话' })).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('.sidebar')).toHaveClass(/sidebar-compact/);
  await page.getByRole('button', { name: '展开侧栏', exact: true }).click();
  await expect(page.locator('.sidebar')).not.toHaveClass(/sidebar-compact/);
  report.checks.push('sidebar-collapse-expand-rapid-persistence');

  await page.getByRole('button', { name: '设置', exact: true }).click();
  await expect(page.getByRole('heading', { name: '设置', exact: true })).toBeVisible();
  await page.getByLabel('模型名称', { exact: true }).fill('e2e-local-persisted');
  await page.getByRole('button', { name: '保存 DeepSeek 设置' }).click();
  await expect.poll(() => page.evaluate(async () => (await window.paperAgent.config()).model)).toBe('e2e-local-persisted');
  const publicConfig = await page.evaluate(() => window.paperAgent.config());
  assert.equal('api_key' in publicConfig, false);
  assert.equal('openalex_api_key' in publicConfig, false);
  assert.equal(publicConfig.api_key_configured, true);
  assert.equal(publicConfig.openalex_api_key_configured, true);
  const secretInputs = page.getByPlaceholder('输入新密钥');
  for (let i = 0; i < await secretInputs.count(); i++) assert.equal(await secretInputs.nth(i).inputValue(), '');
  await page.reload();
  await expect(page.getByRole('heading', { name: '设置', exact: true })).toBeVisible();
  await expect(page.getByLabel('模型名称', { exact: true })).toHaveValue('e2e-local-persisted');
  assert.equal((await page.evaluate(() => Object.keys(localStorage))).every(key => /layout/.test(key)), true);
  report.checks.push('settings-persist-reload-keys-write-only-layout-only-storage');

  // Appearance is independent of backend/model configuration and persists locally.
  const themeGroup = page.getByRole('radiogroup', { name: '界面主题', exact: true });
  await expect(themeGroup.getByRole('radio', { name: '跟随系统', exact: true })).toBeChecked();
  await page.emulateMedia({ colorScheme: 'dark' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await themeGroup.getByRole('radio', { name: '浅色', exact: true }).check();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--canvas').trim()), '#f5f3ee');
  await page.emulateMedia({ colorScheme: 'light' });
  await themeGroup.getByRole('radio', { name: '深色', exact: true }).check();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--canvas').trim()), '#161513');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(themeGroup.getByRole('radio', { name: '深色', exact: true })).toBeChecked();
  await themeGroup.getByRole('radio', { name: '跟随系统', exact: true }).check();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.emulateMedia({ colorScheme: 'dark' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.emulateMedia({ colorScheme: null });
  assert.equal((await page.evaluate(() => window.paperAgent.config())).model, 'e2e-local-persisted');
  report.checks.push('theme-system-light-dark-overrides-persistence-and-model-config-isolation');

  // Closing the dock must preserve the mounted panel state, not discard edits.
  const baseUrlInput = page.getByLabel('API Base URL', { exact: true });
  const originalBaseUrl = await baseUrlInput.inputValue();
  await baseUrlInput.fill('https://example.org/v1');
  await page.locator('.right-dock [aria-label="收起资料 Dock"]').click();
  await expect(page.locator('.workspace-no-dock')).toBeVisible();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.locator('.conversation-header [aria-label="打开资料 Dock"]').click();
  await expect(page.locator('.workspace-no-dock')).toHaveCount(0);
  await expect(baseUrlInput).toHaveValue('https://example.org/v1');
  await baseUrlInput.fill(originalBaseUrl);
  report.checks.push('dock-close-reopen-preserves-panel-state');

  await page.getByRole('tab', { name: /^论文。/ }).click();
  await page.getByRole('button', { name: '添加 Dock 面板' }).click();
  await page.getByRole('menuitem', { name: '研究报告' }).click();
  await page.getByRole('tab', { name: /^论文。/ }).click();
  await page.getByRole('button', { name: '分成两个组', exact: true }).click();
  assert.equal(await page.locator('.dock-group').count(), 2);
  const reportTab = page.getByRole('tab', { name: /^研究报告。/ });
  await reportTab.focus();
  await page.keyboard.press('Alt+ArrowUp');
  assert.equal(await page.locator('[aria-label="第二组资料"] [role="tab"]').count(), 2);
  await page.getByRole('button', { name: '关闭 研究报告', exact: true }).click();
  await expect(page.getByRole('tab', { name: /^研究报告。/ })).toHaveCount(0);
  await page.getByRole('button', { name: '添加 Dock 面板' }).click();
  await page.getByRole('menuitem', { name: '研究报告' }).click();
  await expect(page.getByRole('tab', { name: /^研究报告。/ })).toBeVisible();
  await page.reload();
  await expect(page.locator('.dock-group')).toHaveCount(2);
  await page.getByRole('button', { name: '合并分组', exact: true }).click();
  await page.getByRole('tab', { name: /^研究报告。/ }).focus();
  const tabOrder = await page.locator('.dock-tab').allTextContents();
  await page.keyboard.press('Control+Shift+ArrowLeft');
  assert.notDeepEqual(await page.locator('.dock-tab').allTextContents(), tabOrder);
  const separator = page.getByRole('separator', { name: '调整资料 Dock 宽度' });
  const before = Number(await separator.getAttribute('aria-valuenow'));
  await separator.focus();
  await page.keyboard.press('ArrowLeft');
  assert.ok(Number(await separator.getAttribute('aria-valuenow')) > before);
  report.checks.push('dock-split-move-close-reopen-persist-resize');

  // Many tabs must remain legible and reachable through scroll/overflow.
  await page.getByRole('button', { name: '添加 Dock 面板' }).click();
  await page.getByRole('menuitem', { name: '执行记录' }).click();
  await expect(page.locator('.dock-overflow-trigger')).toBeVisible();
  const slots = page.locator('.dock-tab-slot');
  assert.ok(await slots.count() >= 4);
  for (let index = 0; index < await slots.count(); index += 1) {
    const slot = await slots.nth(index).boundingBox();
    assert.ok(slot && slot.width >= 100, 'dock tab slot collapsed below readable width');
  }
  await page.getByRole('button', { name: '更多资料标签' }).click();
  await expect(page.getByRole('menu')).toBeVisible();
  await page.getByRole('menuitemradio', { name: /执行记录/ }).click();
  await expect(page.locator('.activity-view, .panel-empty')).toBeVisible();
  report.checks.push('dock-tabs-overflow-scroll-menu');

  // Submit an explicitly labeled fixture through the real Electron bridge and MCP server.
  const job = await page.evaluate(() => window.paperAgent.sendMessage({
    text: 'Agent memory research (fixture)', query: 'agent memory', action: 'research',
    top_k: 1, mode: 'demo', execution_mode: 'concurrent',
  }));
  report.job_id = job.job_id;
  const activeBeforeReload = await page.evaluate(id => window.paperAgent.job(id), job.job_id);
  report.status_before_reload = activeBeforeReload.status;
  if (dev) {
    await page.evaluate(async () => {
      const { createHotContext } = await import('/@vite/client');
      window.__hmrApplied = false;
      window.__hmrExpected = '/src/app/theme/theme.css';
      createHotContext('/desktop-e2e-observer').on('vite:afterUpdate', value => {
        if (value.updates.some(update => update.path === window.__hmrExpected || update.acceptedPath === window.__hmrExpected)) window.__hmrApplied = true;
      });
    });
    const css = path.join(root, 'frontend/src/app/theme/tokens.css');
    const previous = await stat(css);
    try {
      await utimes(css, previous.atime, new Date());
      await expect.poll(() => page.evaluate(() => window.__hmrApplied), { timeout: 15_000 }).toBe(true);
    }
    finally { await utimes(css, previous.atime, previous.mtime); }
    const component = path.join(root, 'frontend/src/app/layout/components/WorkspaceShell.tsx');
    const componentTime = await stat(component);
    await page.evaluate(() => {
      window.__hmrApplied = false;
      window.__hmrExpected = '/src/app/layout/components/WorkspaceShell.tsx';
    });
    try {
      await utimes(component, componentTime.atime, new Date());
      await expect.poll(() => page.evaluate(() => window.__hmrApplied), { timeout: 15_000 }).toBe(true);
      await expect(page.getByRole('main', { name: '研究对话' })).toBeVisible();
    } finally { await utimes(component, componentTime.atime, componentTime.mtime); }
    assert.equal((await page.evaluate(() => window.paperAgent.info())).pid, info.pid);
    report.checks.push('vite-csp-fast-refresh-hmr-same-daemon');
  }
  await page.reload();
  await expect(page.locator('.conversation-row').first()).toBeVisible({ timeout: 30_000 });
  await expect.poll(async () => (await page.evaluate(() => window.paperAgent.conversations())).length).toBe(1);
  await expect.poll(async () => (await page.evaluate(id => window.paperAgent.job(id), job.job_id)).status,
    { timeout: 120_000, intervals: [250, 500, 1000] }).toMatch(/^(completed|partial)$/);
  const finished = await page.evaluate(id => window.paperAgent.job(id), job.job_id);
  assert.ok(finished.result?.report);
  assert.ok(finished.artifacts.length > 0);
  assert.equal(finished.conversation_id, job.conversation_id);
  assert.ok(finished.events.length > 5);
  assert.equal(new Set(finished.events.map(e => e.seq)).size, finished.events.length);
  report.result_status = finished.status;
  report.event_count = finished.events.length;
  report.artifact_count = finished.artifacts.length;
  await page.evaluate(id => new Promise((resolve, reject) => {
    let timer;
    const off = window.paperAgent.subscribeJob(id, 0, value => {
      if (value.error) { clearTimeout(timer); off(); reject(new Error(value.error)); }
      if (value.done) { clearTimeout(timer); off(); resolve(true); }
    });
    timer = setTimeout(() => { off(); reject(new Error('SSE replay timed out')); }, 15_000);
  }), job.job_id);
  report.checks.push('actual-mcp-fixture-job-reload-no-duplicate-sse-replay');

  await page.getByRole('tab', { name: /^执行记录。/ }).click();
  const executionLog = page.locator('details.execution-log');
  await expect(executionLog).toBeVisible();
  assert.equal(await executionLog.evaluate(element => element.open), false);
  await executionLog.locator(':scope > summary').click();
  await expect(page.locator('details.activity-card').first()).toBeVisible();
  report.checks.push('execution-log-default-collapsed-expandable-detail-cards');

  await page.getByRole('tab', { name: /^论文。/ }).click();
  await expect(page.locator('.paper-row').first()).toBeVisible({ timeout: 15_000 });
  await page.getByRole('button', { name: /^阅读论文：/ }).first().click();
  await expect(page.locator('.reader-view')).toBeVisible();
  await page.getByRole('tab', { name: '原文 Markdown', exact: true }).click();
  await expect(page.getByRole('tabpanel', { name: '原文 Markdown', exact: true })).toContainText('synthetic development fixture');
  await page.getByRole('tab', { name: 'PDF 原文', exact: true }).click();
  await expect(page.locator('canvas.pdf-canvas')).toBeVisible();
  await page.getByRole('tab', { name: /^研究报告。/ }).click();
  await expect(page.locator('.report-view')).toBeVisible();
  const artifact = finished.artifacts.find(a => a.kind === 'report' || (a.kind === 'markdown' && /report/i.test(a.name)));
  assert.ok(artifact);
  const text = await page.evaluate(({ j, a }) => window.paperAgent.readArtifact(j, a), { j: job.job_id, a: artifact.artifact_id });
  assert.ok(text.text.length > 20);
  // v0.2.2 second iteration: the comparison section must stay readable Markdown
  // in the stored artifact instead of leaking raw JSON records.
  assert.match(text.text, /## 论文比较/);
  assert.ok(text.text.includes(finished.result.summaries[0].title));
  assert.ok(!text.text.includes('[demo:001]'), 'reading artifact should cite the paper title');
  assert.ok(!text.text.includes('"paper_ids"'), 'report comparison must not leak raw JSON');
  await app.evaluate(({ dialog }, exportPath) => {
    dialog.showSaveDialog = async () => ({ canceled: false, filePath: exportPath });
  }, path.join(output, 'exported-report.md'));
  await page.getByRole('button', { name: '导出', exact: true }).click();
  await expect.poll(async () => { try { return (await readFile(path.join(output, 'exported-report.md'), 'utf8')).length; } catch { return 0; } }).toBeGreaterThan(20);
  await page.getByRole('button', { name: '复制', exact: true }).click();
  assert.ok((await app.evaluate(({ clipboard }) => clipboard.readText())).length > 20);
  const blocked = await page.evaluate(async () => {
    try { await window.paperAgent.openExternal('file:///C:/Windows/system.ini'); return false; } catch { return true; }
  });
  assert.equal(blocked, true);
  report.checks.push('paper-detail-report-artifact-export-copy-url-validation');

  // Readable report body and the persisted fixture scope remain available
  // without restoring the removed diagnostic/status chip stack.
  await expect(page.locator('.report-state-note, .report-view .delivery-badge')).toHaveCount(0);
  assert.equal(finished.result.summaries[0].summary_scope, 'demo_fixture');
  const renderedReport = await page.locator('.report-markdown').innerText();
  assert.match(renderedReport, /论文比较/);
  assert.match(renderedReport, /比较 1/);
  assert.ok(renderedReport.includes('synthetic fixtures only'));
  assert.ok(!renderedReport.includes('"paper_ids"'), 'rendered comparison must not leak raw JSON');
  assert.equal(finished.result?.metrics?.report_valid, true);
  await page.getByRole('tab', { name: /^论文。/ }).click();
  const finalRows = page.locator('.paper-section[aria-labelledby="effective-papers-title"] .paper-row');
  await expect(finalRows).toHaveCount(1);
  assert.equal(await page.locator('.paper-section[aria-labelledby="effective-papers-title"] .paper-rank').innerText(), '1');
  const paperTags = await page.locator('.paper-section[aria-labelledby="effective-papers-title"] .paper-material').first().innerText();
  assert.ok(paperTags.includes('演示样例'), `summary scope tag missing: ${paperTags}`);
  assert.ok(paperTags.includes('PDF 原文可读'), `PDF availability missing: ${paperTags}`);
  // Restore the report tab so this block leaves the dock as it found it.
  await page.getByRole('tab', { name: /^研究报告。/ }).click();
  await expect(page.locator('.report-view')).toBeVisible();
  report.checks.push('report-validity-scope-readable-comparison-stable-numbering');
  await page.getByRole('button', { name: '添加 Dock 面板' }).click();
  await page.getByRole('menuitem', { name: '性能数据' }).click();
  await expect(page.getByRole('region', { name: '运行时间', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: '运行时间', exact: true })).toContainText('并行');
  await expect(page.getByRole('region', { name: '语义检索 Top-1', exact: true })).toContainText('暂无相关性记录');
  await expect(page.getByRole('region', { name: '任务 Agent 筛选 Top-1', exact: true })).toContainText(finished.result.selected_papers[0].title);
  report.checks.push('current-task-top-k-relevance-and-duration');
  await page.getByRole('button', { name: '新建会话', exact: true }).click();
  const finishedPaperTitle = finished.result.selected_papers[0].title;
  await page.locator('.dock-tab').filter({ hasText: finishedPaperTitle }).first().click();
  await expect(page.locator('.reader-view')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText('该论文不在所属任务结果中')).toHaveCount(0);
  report.checks.push('paper-tab-restores-own-job-after-conversation-switch');
  await page.getByRole('tab', { name: /^研究报告。/ }).click();
  await parallelSwitch.check();
  await page.screenshot({ path: path.join(output, 'desktop-wide.png') });
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1024, 720));
  await expect(page.locator('.workspace-stacked')).toBeVisible();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await parallelSwitch.uncheck();
  await page.screenshot({ path: path.join(output, 'desktop-compact.png') });
  await page.getByRole('button', { name: '收起侧栏', exact: true }).click();
  await expect(page.locator('.sidebar')).toHaveClass(/sidebar-compact/);
  await expect.poll(async () => (await page.locator('.sidebar').boundingBox())?.width).toBe(58);
  await expect(page.getByRole('button', { name: '展开侧栏', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '展开侧栏', exact: true }).click();
  report.checks.push('tablet-sidebar-toggle');
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1440, 960));
  await page.getByRole('button', { name: '新建会话', exact: true }).click();
  const draft = page.getByRole('textbox', { name: '研究问题', exact: true });
  // IME composition must never submit a job.
  await expect(draft).toBeVisible();
  const initialComposerHeight = await draft.evaluate(element => element.getBoundingClientRect().height);
  await draft.fill(['第一行','第二行','第三行','第四行','第五行','第六行'].join('\n'));
  await expect.poll(() => draft.evaluate(element => element.getBoundingClientRect().height)).toBeGreaterThan(initialComposerHeight);
  assert.ok((await draft.evaluate(element => element.getBoundingClientRect().height)) <= 191);
  await expect(page.locator('#composer-count')).toContainText('/12000');
  report.checks.push('composer-autogrow-character-count');
  await draft.fill('输入法组合测试');
  await draft.dispatchEvent('keydown', { key: 'Enter', code: 'Enter', isComposing: true });
  assert.equal((await page.evaluate(() => window.paperAgent.conversations())).length, 1);
  const cancelled = await page.evaluate(async () => {
    const job = await window.paperAgent.sendMessage({ text: 'Cancellation fixture', action: 'research', top_k: 10, mode: 'demo' });
    return window.paperAgent.cancelJob(job.job_id);
  });
  await expect.poll(async () => (await page.evaluate(id => window.paperAgent.job(id), cancelled.job_id)).status).toBe('cancelled');
  report.checks.push('ime-enter-no-submit-explicit-job-cancel');
  assert.deepEqual(errors, []);
  report.checks.push('compact-viewport-no-overflow-no-renderer-errors');
  privateEndpoint = JSON.parse(await readFile(path.join(profile, 'service/endpoint.json'), 'utf8'));
} catch (error) {
  failure = error;
  report.failure = error.stack;
  try { await (await app.firstWindow()).screenshot({ path: path.join(output, 'failure.png') }); } catch { /* App may have exited. */ }
} finally {
  await app.close();
  await devServer?.close();
  try {
    privateEndpoint ??= JSON.parse(await readFile(path.join(profile, 'service/endpoint.json'), 'utf8'));
    const headers = { Authorization: `Bearer ${privateEndpoint.token}` };
    const response = await fetch(`${privateEndpoint.url}/api/service`, { headers });
    assert.equal(response.status, 200);
    report.checks.push('daemon-survives-window-close');
    const stop = await fetch(`${privateEndpoint.url}/api/service/shutdown`, { method: 'POST', headers });
    assert.equal(stop.status, 200);
  } catch (error) { report.cleanup_error = error.message; }
  await writeFile(path.join(output, 'result.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ output, checks: report.checks, failure: Boolean(failure), ...('cleanup_error' in report ? { cleanup_error: report.cleanup_error } : {}) }, null, 2));
}
if (failure) throw failure;
