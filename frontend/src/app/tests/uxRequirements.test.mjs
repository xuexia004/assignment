import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import Module from 'node:module';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { buildSync } from 'esbuild';

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = relative => fs.readFileSync(path.join(appRoot, relative), 'utf8');

function load(relative) {
  const filename = path.join(appRoot, relative);
  const built = buildSync({
    entryPoints: [filename], bundle: true, platform: 'node', format: 'cjs',
    write: false, logLevel: 'silent',
  });
  const loaded = new Module(filename);
  loaded.filename = filename;
  loaded.paths = Module._nodeModulePaths(path.dirname(filename));
  loaded._compile(built.outputFiles[0].text, filename);
  return loaded.exports;
}

test('research brief keeps at most three supported findings without placeholders', () => {
  const { supportedFindings } = load('utils/researchBrief.ts');
  const finding = statement => ({ statement, evidence: [{ paper_id: 'fixture-paper' }] });
  const brief = {
    findings: [finding('one'), { statement: 'unsupported', evidence: [] }, finding('two'), finding('three'), finding('four')],
    key_papers: [], next_step: { action: 'verify' }, limitations: [],
  };
  assert.deepEqual(supportedFindings(brief).map(item => item.statement), ['one', 'two', 'three']);
  assert.deepEqual(supportedFindings({ ...brief, findings: [finding('only one')] }).length, 1);
  assert.deepEqual(supportedFindings({ ...brief, findings: [
    finding('作者报告基线 LLM 降低幻觉（f12）'), finding('基线LLM降低幻觉'), finding('基线LLM未降低幻觉'),
  ] }).map(item => item.statement), ['作者报告基线 LLM 降低幻觉（f12）', '基线LLM未降低幻觉']);
});

test('preferred paper count and the parallel optimization switch remain explicit in the composer', () => {
  const conversation = source('layout/components/conversation/Conversation.tsx');
  const countSelect = source('layout/components/conversation/PaperCountSelect.tsx');
  assert.match(countSelect, /\[3, 5, 8\]/);
  assert.match(countSelect, /aria-label="精选论文数量"/);
  assert.match(conversation, /高级设置/);
  assert.match(conversation, /role="switch" aria-label="并行优化"/);
  assert.match(conversation, /event.target.checked \? 'parallel' : 'sequential'/);
  assert.doesNotMatch(conversation, /aria-label="研究执行方式"/);
  const store = source('stores/workspaceStore.ts');
  assert.match(store, /executionMode: 'sequential'/);
  const { requestedExecutionMode } = load('utils/researchBrief.ts');
  assert.equal(requestedExecutionMode(undefined), 'sequential');
  assert.equal(requestedExecutionMode('auto'), 'sequential');
  assert.equal(requestedExecutionMode('parallel'), 'parallel');
  assert.equal(requestedExecutionMode('concurrent'), 'parallel');
});

test('paper title citations cover old reports without rewriting URLs, code or machine references', () => {
  const { paperTitleCitations } = load('utils/paperCitations.ts');
  const papers = [
    { paper_id: 'arxiv:2604.17293', title: 'Beyond I Don’t Know' },
    { paper_id: 'openalex:W7148253449', title: 'Truth-Aligned Uncertainty' },
  ];
  const input = '[arxiv:2604.17293] 对比 openalex:W7148253449。\n\n'
    + '- [arxiv:2604.17293] Beyond I Don’t Know\n\n'
    + '[arxiv:2604.17293](https://example.org/arxiv:2604.17293)\n\n'
    + '`arxiv:2604.17293`\n\n[arxiv:2604.172930]';
  const output = paperTitleCitations(input, papers);
  assert.ok(output.includes('《Beyond I Don’t Know》 对比 Truth-Aligned Uncertainty'));
  assert.ok(output.includes('- 《Beyond I Don’t Know》\n'));
  assert.ok(!output.includes('《Beyond I Don’t Know》 Beyond I Don’t Know'));
  assert.ok(output.includes('[Beyond I Don’t Know](https://example.org/arxiv:2604.17293)'));
  assert.ok(output.includes('`arxiv:2604.17293`'));
  assert.ok(output.includes('[arxiv:2604.172930]'));
  assert.equal(papers[0].paper_id, 'arxiv:2604.17293');
  const { readableReportMarkdown } = load('utils/reportMarkdown.ts');
  const dynamic = '# 如何选择拒答机制\n\n## 决策建议\n\n先比较风险控制条件。';
  assert.equal(readableReportMarkdown(dynamic, [{ paper_ids: ['arxiv:2604.17293'], method: 'some metadata' }], papers), dynamic);
});

test('all three delivery states stay separate from execution completion', () => {
  const { deliveryStatusFor, deliveryStatusLabels } = load('utils/researchBrief.ts');
  assert.deepEqual(Object.keys(deliveryStatusLabels).sort(), ['no_result', 'preliminary_notes', 'result']);
  assert.equal(deliveryStatusFor({ delivery_status: 'no_result', status: 'completed' }), 'no_result');
  assert.equal(deliveryStatusFor({ status: 'completed' }, { delivery_status: 'preliminary_notes' }), 'preliminary_notes');
  assert.equal(deliveryStatusFor({ status: 'completed' }), null);
});

test('execution log is folded by default and merges identical errors with details retained', () => {
  const { groupExecutionErrors } = load('utils/executionLog.ts');
  const duplicate = { code: 'invalid_model_output', stage: 'summary', message: 'fixture invalid output' };
  const groups = groupExecutionErrors([duplicate, { ...duplicate }, { ...duplicate, message: 'different' }]);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].count, 2);
  assert.equal(groups[0].items.length, 2);
  const activity = source('layout/components/activity/ActivityPanel.tsx');
  assert.match(activity, /<details className="execution-log">/);
  assert.doesNotMatch(activity, /<details className="execution-log" open/);
  assert.match(activity, /相同错误 ×/);
});

test('paper reader exposes three peer views and keeps PDF navigation controls', () => {
  const reader = source('layout/components/paper-reader/PaperReader.tsx');
  assert.match(reader, /研究笔记/);
  assert.match(reader, /PDF 原文/);
  assert.match(reader, /原文 Markdown/);
  const viewer = source('layout/components/paper-reader/PdfViewer.tsx');
  for (const label of ['上一页', '下一页', 'PDF 页码', '放大 PDF', '缩小 PDF', '搜索 PDF 原文', '扩大 PDF 阅读']) {
    assert.ok(viewer.includes(label), label);
  }
  assert.match(viewer, /readPdfArtifact/);
  assert.doesNotMatch(viewer, /readArtifact\(/);
  assert.match(viewer, /当前桌面运行时没有 readPdfArtifact 接口/);
});

test('PDF page, zoom, search and evidence location use bounded physical-page parameters', () => {
  const { canLocatePdfEvidence, normalizePdfPage, normalizePdfZoom, pdfViewerUrl } = load('utils/pdf.ts');
  assert.equal(normalizePdfPage(-4), 1);
  assert.equal(normalizePdfPage('3'), 3);
  assert.equal(normalizePdfZoom(999), 200);
  const url = pdfViewerUrl('blob:fixture', 4, 125, 'evidence term');
  assert.match(url, /page=4/);
  assert.match(url, /zoom=125/);
  assert.match(url, /search=evidence\+term/);
  assert.equal(canLocatePdfEvidence({ paper_id: 'p', pdf_artifact_id: 'a', page_number: 2 }), true);
  assert.equal(canLocatePdfEvidence({ paper_id: 'p', pdf_artifact_id: 'a', page_number: 0 }), false);
  assert.equal(canLocatePdfEvidence({ paper_id: 'p', page_number: 2 }), false);
});

test('canonical report artifact is the single report reading target', () => {
  const { reportArtifactFor } = load('utils/researchBrief.ts');
  const artifacts = [
    { artifact_id: 'other', kind: 'report', name: 'other.md' },
    { artifact_id: 'canonical', kind: 'report', name: 'canonical.md' },
  ];
  assert.equal(reportArtifactFor({ brief: { report_artifact_id: 'canonical' }, report: { report_artifact_id: 'other' } }, artifacts).artifact_id, 'canonical');
  const conversation = source('layout/components/conversation/Conversation.tsx');
  assert.match(conversation, /brief-report-link|ResearchBriefView/);
  assert.doesNotMatch(conversation, /<MarkdownContent text=\{result\.report/);
});

test('recovery covers pending and partial success without submit or cancel on cleanup', () => {
  const { isJobActive, isJobFinished } = load('utils/status.ts');
  assert.equal(isJobActive('pending'), true);
  assert.equal(isJobFinished('partial_success'), true);
  const recovery = source('hooks/useJobRecovery.ts');
  assert.match(recovery, /import\.meta\.hot\.dispose/);
  assert.match(recovery, /subscribeJob/);
  assert.match(recovery, /refreshJob/);
  assert.doesNotMatch(recovery, /sendMessage|cancelJob|cancelActiveJob/);
});

test('task metrics keep semantic ranks, Agent scores and wall time distinct', () => {
  const { taskRankings, taskTotalSeconds } = load('utils/taskMetrics.ts');
  const papers = [
    { paper_id: 'a', initial_score: 0.99, ranking_details: { cosine_similarity: 0.2 } },
    { paper_id: 'b', initial_score: 0.1, ranking_details: { cosine_similarity: 0.8 } },
    { paper_id: 'missing', initial_score: 1, ranking_details: {} },
    { paper_id: 'zero', ranking_details: { cosine_similarity: 0 } },
  ];
  const result = { papers, top_k: 2, selected_papers: [
    { paper_id: 'a', rank: 2, final_score: 0 },
    { paper_id: 'b', rank: 1, final_score: 0.9 },
  ], metrics: { main_agent_wall_seconds: 12 }, timings: [
    { stage: 'search', duration_seconds: 4 },
    { stage: 'total', duration_seconds: 10 },
  ] };
  const metrics = taskRankings(result);
  assert.deepEqual(metrics.semantic.map(paper => paper.paper_id), ['b', 'a']);
  assert.deepEqual(metrics.selected.map(paper => paper.paper_id), ['b', 'a']);
  assert.equal(metrics.semanticMean, 0.5);
  assert.equal(metrics.selectedMean, 0.45);
  assert.equal(taskTotalSeconds(result), 12);
  assert.equal(taskTotalSeconds({ ...result, metrics: {} }), 10);
  assert.equal(taskTotalSeconds({ ...result, metrics: {}, timings: [] }), null);
  assert.deepEqual(papers.map(paper => paper.paper_id), ['a', 'b', 'missing', 'zero']);
  assert.equal(taskRankings({ ...result, papers: [papers[2]] }).semanticMean, null);
});
