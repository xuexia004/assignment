/** Integrated UI checks; run only through mock-desktop.mjs after Agents 8/9 land. */
import { expect } from '@playwright/test';
import assert from 'node:assert/strict';

async function openScenario(page, scenario) {
  const row = page.locator('.conversation-row').filter({ hasText: `Fixture ${scenario}` }).first();
  await expect(row).toBeVisible({ timeout: 15_000 });
  await row.click();
}

async function executionLog(page) {
  let log = page.locator('details.execution-log').first();
  if (await log.count() === 0) {
    await page.getByRole('button', { name: '添加 Dock 面板' }).click();
    await page.getByRole('menuitem', { name: '执行记录', exact: true }).click();
    log = page.locator('details.execution-log').first();
  }
  await expect(log).toBeVisible();
  if (await log.evaluate(element => element.open)) await log.locator(':scope > summary').click();
  assert.equal(await log.evaluate(element => element.open), false, 'Execution log must start collapsed.');
  return log;
}

export async function runMockUiChecks(page, jobs) {
  await openScenario(page, 'success');
  await expect(page.getByText('Fixture finding 1', { exact: true })).toBeVisible();
  await expect(page.getByText('Fixture finding 3', { exact: true })).toBeVisible();
  await expect(page.locator('.research-brief .delivery-badge')).toHaveCount(0);
  await expect(page.getByText('Inspect fixture page two.', { exact: true })).toBeVisible();
  const successLog = await executionLog(page);
  await successLog.locator(':scope > summary').click();
  assert.equal(await successLog.evaluate(element => element.open), true);

  await openScenario(page, 'partial');
  await expect(page.locator('.research-brief .delivery-badge, .brief-state-note')).toHaveCount(0);
  await expect(page.getByText('Fixture finding 1', { exact: true })).toBeVisible();
  await expect(page.getByText('Fixture finding 2', { exact: true })).toHaveCount(0);
  await expect(page.locator('.brief-limitations')).toHaveCount(0);
  await executionLog(page);

  await openScenario(page, 'no_result');
  await expect(page.locator('.research-brief .delivery-badge')).toHaveCount(0);
  await expect(page.getByText('Provide readable source PDFs.', { exact: true })).toBeVisible();
  await expect(page.getByText('Fixture finding 1', { exact: true })).toHaveCount(0);

  await openScenario(page, 'retry_recovered');
  await expect(page.locator('.research-brief .delivery-badge')).toHaveCount(0);
  await expect(page.getByRole('main').getByText(/invalid_model_output/)).toHaveCount(0);
  const recoveredLog = await executionLog(page);
  await recoveredLog.locator(':scope > summary').click();
  await recoveredLog.locator('.event-section').filter({ has: page.getByRole('heading', { name: /^服务端事件/ }) })
    .locator('details.activity-card').first().locator(':scope > summary').click();
  await expect(recoveredLog.getByText(/invalid_model_output/).first()).toBeVisible();
  await expect(recoveredLog.getByText(/retry recovered/i)).toBeVisible();

  await openScenario(page, 'retry_exhausted');
  await expect(page.locator('.research-brief .delivery-badge, .brief-state-note')).toHaveCount(0);
  await expect(page.locator('.brief-limitations')).toHaveCount(0);
  const exhaustedLog = await executionLog(page);
  await exhaustedLog.locator(':scope > summary').click();
  await expect(exhaustedLog.getByText(/相同错误\s*×\s*2/)).toBeVisible();
  const repeatedError = exhaustedLog.locator('details.activity-error').first();
  await repeatedError.locator(':scope > summary').click();
  await expect(repeatedError.getByText(/invalid_model_output/).first()).toBeVisible();

  // Verify peer reading views against registered local artifacts.
  await openScenario(page, 'success');
  await page.getByRole('tab', { name: /^论文。/ }).click();
  await page.getByRole('button', { name: /^阅读论文：Deterministic fixture paper/ }).click();
  await page.getByRole('tab', { name: '研究笔记', exact: true }).click();
  await expect(page.getByRole('heading', { name: '证据与原文定位' })).toHaveCount(0);
  await page.getByRole('tab', { name: '原文 Markdown', exact: true }).click();
  await expect(page.getByRole('tabpanel', { name: '原文 Markdown', exact: true })).toContainText('Original extracted paper text.');
  await page.getByRole('tab', { name: 'PDF 原文', exact: true }).click();
  await expect(page.getByRole('tab', { name: 'PDF 原文', exact: true })).toHaveAttribute('aria-selected', 'true');
  const pageNumber = page.getByRole('spinbutton', { name: 'PDF 页码', exact: true });
  await expect(pageNumber).toHaveValue('1');
  await expect(page.locator('canvas.pdf-canvas')).toBeVisible();
  await page.getByRole('button', { name: '下一页', exact: true }).click();
  await expect(pageNumber).toHaveValue('2');
  await page.getByRole('button', { name: '下一页', exact: true }).click();
  await expect(pageNumber).toHaveValue('3');
  await page.getByRole('button', { name: '放大 PDF', exact: true }).click();
  await expect(page.locator('.pdf-zoom-controls')).toContainText('125%');
  const search = page.getByRole('textbox', { name: '搜索 PDF 原文', exact: true });
  await search.fill('FIXTURE PAGE THREE');
  await page.getByRole('button', { name: '查找', exact: true }).click();
  await expect(page.locator('.pdf-search-note')).toContainText('找到 1 处');

  await page.getByRole('button', { name: '添加 Dock 面板' }).click();
  await page.getByRole('menuitem', { name: '性能数据', exact: true }).click();
  await expect(page.getByRole('region', { name: '语义检索 Top-3', exact: true })).toContainText('0.750');
  await expect(page.getByRole('region', { name: '任务 Agent 筛选 Top-3', exact: true })).toContainText('1.000');
  await expect(page.getByRole('region', { name: '运行时间', exact: true })).toContainText('5.0 秒');

  assert.equal(jobs.success.result.brief.findings[0].evidence[0].page_number, 2);
}
