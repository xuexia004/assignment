import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUp, BookOpen, Copy, Download, FileText, FolderOpen } from 'lucide-react';
import type { Artifact, ContractError, PipelineResult } from '../../../types/contracts';
import { desktopApi } from '../../../services/desktopApi';
import { useWorkspaceState, workspaceStore } from '../../../stores/workspaceStore';
import { errorLabel } from '../../../utils/labels';
import { readableReportMarkdown } from '../../../utils/reportMarkdown';
import { reportArtifactFor } from '../../../utils/researchBrief';
import { paperTitleCitations } from '../../../utils/paperCitations';
import { visibleReportDocument, type LoadedReportDocument } from '../../../utils/resultPresentation';
import { MarkdownContent } from '../common/MarkdownContent';

export function ResearchReport() {
  const { activeJob, resultJob, conversation } = useWorkspaceState();
  const job = resultJob ?? activeJob;
  const report = job?.result?.report ?? conversation?.result?.report ?? null;
  const result = job?.result ?? conversation?.result ?? null;
  const artifacts = job?.artifacts ?? [];
  const reportArtifact = reportArtifactFor(result, artifacts);
  const reportKey = job && reportArtifact ? `${job.job_id}:${reportArtifact.artifact_id}` : null;
  const [artifactDocument, setArtifactDocument] = useState<LoadedReportDocument | null>(null);
  const [exported, setExported] = useState<{ key: string; path: string } | null>(null);
  const inlineMarkdown = report?.markdown?.trim() ? report.markdown : '';
  const { markdown, truncated, loading, error } = visibleReportDocument(inlineMarkdown, reportKey, artifactDocument);
  const exportedPath = reportKey && exported?.key === reportKey ? exported.path : null;
  const contentRef = useRef<HTMLDivElement | null>(null);
  const displayMarkdown = useMemo(
    () => readableReportMarkdown(markdown, report?.comparison, [
      ...(result?.papers ?? []), ...(result?.selected_papers ?? []), ...(report?.references ?? []),
    ]),
    [markdown, report?.comparison, report?.references, result?.papers, result?.selected_papers],
  );
  const hasSubstance = reportHasSubstance(report, displayMarkdown);

  useEffect(() => {
    if (!job || !reportArtifact || !reportKey || inlineMarkdown) return;
    let current = true;
    setArtifactDocument({ key: reportKey, text: '', truncated: false, error: null, loading: true });
    void Promise.resolve().then(() => desktopApi().readArtifact(job.job_id, reportArtifact.artifact_id)).then(value => {
      if (current) setArtifactDocument({ key: reportKey, text: value.text, truncated: value.truncated, error: null, loading: false });
    }).catch(reason => {
      if (current) setArtifactDocument({ key: reportKey, text: '', truncated: false,
        error: reason instanceof Error ? reason.message : String(reason), loading: false });
    });
    return () => { current = false; };
  }, [reportKey, inlineMarkdown]);

  const copyReport = async () => {
    if (!displayMarkdown) return;
    try {
      await desktopApi().copyText(displayMarkdown);
      workspaceStore.setNotice('研究报告已复制。');
    } catch (reason) {
      workspaceStore.setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  const exportReport = async () => {
    if (!job || !reportArtifact) return;
    try {
      const response = await desktopApi().exportArtifact(job.job_id, reportArtifact.artifact_id);
      if (reportKey) setExported(response.cancelled || !response.path ? null : { key: reportKey, path: response.path });
      workspaceStore.setNotice(response.cancelled ? '已取消导出。' : response.path ? `报告已导出到：${response.path}` : '报告导出完成。');
    } catch (reason) {
      workspaceStore.setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  const openReport = async () => {
    if (!job || !reportArtifact) return;
    try {
      await desktopApi().openArtifact(job.job_id, reportArtifact.artifact_id);
    } catch (reason) {
      workspaceStore.setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  const scrollToHeading = (title: string) => {
    const root = contentRef.current;
    if (!root) return;
    const nodes = Array.from(root.querySelectorAll('h2, h3'));
    const target = nodes.find(node => node.textContent?.trim() === title)
      ?? nodes.find(node => node.textContent?.includes(title));
    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  const scrollToTop = () => {
    const container = contentRef.current?.closest('.dock-panel-content') as HTMLElement | null;
    container?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (!report && !reportArtifact) return <div className="panel-empty"><span><FileText size={21}/></span><strong>研究报告尚未生成</strong><p>研究完成后，可以在这里阅读和导出综合报告。</p></div>;

  const reportErrors = collectReportErrors(result, report);
  const headings = markdownHeadings(displayMarkdown);
  const references = report?.references ?? [];
  const reportLimitations = (report?.limitations ?? []).filter(item =>
    !item.startsWith('部分报告：') && !item.startsWith('报告提示使用了') &&
    !item.startsWith('报告输入超过') && !item.startsWith('完整总结集'));
  return <article className="report-view">
    <header className="report-header"><div><h2>综合研究报告</h2><p>{job?.idea ?? conversation?.title ?? '当前会话'}</p></div><span className="report-file-mark"><FileText size={17}/></span></header>
    {references.length > 0 && <div className="report-status-row"><span className="report-meta">参考 {references.length} 篇论文</span></div>}
    <div className="report-actions">
      <button className="subtle-button" type="button" disabled={!displayMarkdown || !hasSubstance} onClick={() => void copyReport()}><Copy size={14}/>复制</button>
      {reportArtifact && <button className="subtle-button" type="button" onClick={() => void exportReport()}><Download size={14}/>导出</button>}
      {reportArtifact && <button className="subtle-button" type="button" onClick={() => void openReport()}><FolderOpen size={14}/>打开文件</button>}
      {displayMarkdown.length > 6000 && <button className="subtle-button" type="button" onClick={scrollToTop}><ArrowUp size={14}/>回到顶部</button>}
      {reportArtifact && <span className="file-name">{reportArtifact.name}</span>}
    </div>
    {exportedPath && <p className="file-confirmation">已导出：{exportedPath}</p>}
    {headings.length > 1 && hasSubstance && <details className="report-outline">
      <summary>报告目录（{headings.length}）</summary>
      <ol>{headings.map((heading, index) => <li key={`${index}:${heading.title}`} className={`outline-level-${heading.level}`}>
        <button type="button" onClick={() => scrollToHeading(heading.title)}>{heading.title}</button>
      </li>)}</ol>
    </details>}
    {loading ? <div className="reader-loading" aria-live="polite"><span className="skeleton-line wide"/><span className="skeleton-line"/><span className="skeleton-line medium"/></div>
      : !hasSubstance ? <section className="report-unavailable" role="status">
        <strong>暂无可读的综合正文</strong>
        <p>{reportFailureNote(reportErrors, '请查看执行记录，或稍后重新生成。')}</p>
      </section>
        : displayMarkdown ? <div ref={contentRef} className="report-markdown"><MarkdownContent text={displayMarkdown}/></div>
          : report?.overview ? <div className="report-preview"><h3>报告概述</h3><p>{paperTitleCitations(report.overview, references)}</p>{reportLimitations.length ? <ul>{reportLimitations.map(limitation => <li key={limitation}>{paperTitleCitations(limitation, references)}</li>)}</ul> : null}</div>
            : <div className="panel-empty compact-empty"><span><BookOpen size={19}/></span><strong>报告正文未随结果返回</strong><p>请检查活动记录中的报告生成状态和服务端错误。</p></div>}
    {truncated && <div className="inline-warning" role="status">报告超过 2 MiB，当前仅显示服务端返回的截断内容。</div>}
    {error && <div className="inline-error" role="alert">报告读取失败：{error}</div>}
  </article>;
}

function collectReportErrors(result: PipelineResult | null, report: PipelineResult['report']): ContractError[] {
  const values = [...(report?.errors ?? []), ...(result?.errors ?? []).filter(item => item.stage === 'report')];
  const seen = new Set<string>();
  const output: ContractError[] = [];
  for (const item of values) {
    const key = `${item.code ?? ''}:${item.stage ?? ''}:${item.message}`;
    if (seen.has(key)) continue;
    seen.add(key);
    output.push(item);
  }
  return output;
}

function reportHasSubstance(
  report: PipelineResult['report'], markdown: string,
): boolean {
  if (report) {
    const hasFields = Boolean(
      report.overview?.trim() || report.related_work?.length || report.comparison?.length
      || report.researched_questions?.length || report.limitations?.length
      || report.research_directions?.length,
    );
    if (hasFields) return true;
  }
  return markdownHasSubstance(markdown);
}

function markdownHasSubstance(markdown: string): boolean {
  if (!markdown.trim()) return false;
  let insideReferences = false;
  for (const rawLine of markdown.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const heading = /^#{1,6}\s+(.+)$/.exec(line);
    if (heading) {
      const title = heading[1];
      insideReferences = title.includes('参考文献') || title.includes('保留的论文引用');
      continue;
    }
    if (insideReferences) continue;
    if (line.startsWith('研究想法：') || line.startsWith('>')) continue;
    if (['未生成。', '未明确报告。', '未在所读材料中明确报告。'].includes(line)) continue;
    if (line.startsWith('未生成可读的综合正文') || line.startsWith('原因：')) continue;
    return true;
  }
  return false;
}

function reportFailureNote(errors: ContractError[], fallback: string): string {
  const primary = errors.find(item =>
    ['invalid_model_output', 'llm_budget', 'llm_failed', 'persist_failed', 'no_summaries'].includes(item.code ?? ''),
  ) ?? errors[0];
  if (!primary) return fallback || '报告生成失败或没有可读正文；请查看错误与真实产物。';
  return errorLabel(primary);
}

function markdownHeadings(markdown: string): Array<{ level: number; title: string }> {
  const output: Array<{ level: number; title: string }> = [];
  for (const match of markdown.matchAll(/^(#{2,3})\s+(.+)$/gm)) {
    const title = match[2].replace(/[*_`]/g, '').trim();
    if (!title) continue;
    output.push({ level: match[1].length, title });
    if (output.length >= 16) break;
  }
  return output;
}

export function artifactForReport(artifacts: Artifact[]): Artifact | undefined {
  return reportArtifactFor(undefined, artifacts);
}
