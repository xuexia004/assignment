import { useEffect, useState } from 'react';
import { ArrowUpRight, BookOpen, Copy, Download, FileText, FolderOpen, MoreHorizontal } from 'lucide-react';
import type { Artifact, Job } from '../../../types/contracts';
import type { LayoutTab } from '../../../types/ui';
import { desktopApi } from '../../../services/desktopApi';
import { useWorkspaceState, workspaceStore } from '../../../stores/workspaceStore';
import { collectPapers, summaryStatus } from '../../../utils/papers';
import { paperTitleCitations } from '../../../utils/paperCitations';
import { safeHttpUrl } from '../../../utils/format';
import { MarkdownContent } from '../common/MarkdownContent';
import { PdfViewer } from './PdfViewer';

type ReaderView = 'summary' | 'markdown' | 'pdf';

export function PaperReader({ tab }: { tab: LayoutTab }) {
  const { activeJob, resultJob, conversation, loadingConversation } = useWorkspaceState();
  const currentJob = resultJob ?? activeJob;
  const [remoteJob, setRemoteJob] = useState<Job | null>(null);
  const [remoteLoading, setRemoteLoading] = useState(false);
  const [remoteError, setRemoteError] = useState<{ jobId: string; message: string } | null>(null);
  const remoteTargetId = tab.jobId && tab.jobId !== currentJob?.job_id ? tab.jobId : null;
  const job = remoteTargetId ? (remoteJob?.job_id === remoteTargetId ? remoteJob : null) : currentJob;

  useEffect(() => {
    if (!tab.jobId || tab.jobId === currentJob?.job_id) {
      setRemoteJob(null);
      setRemoteError(null);
      setRemoteLoading(false);
      return;
    }
    let cancelled = false;
    setRemoteLoading(true);
    setRemoteError(null);
    void Promise.resolve().then(() => desktopApi().job(tab.jobId!)).then(value => {
      if (!cancelled) setRemoteJob(value);
    }).catch(error => {
      if (!cancelled) setRemoteError({ jobId: tab.jobId!, message: error instanceof Error ? error.message : String(error) });
    }).finally(() => {
      if (!cancelled) setRemoteLoading(false);
    });
    return () => { cancelled = true; };
  }, [tab.jobId, currentJob?.job_id]);

  const collection = collectPapers(job?.result ?? (tab.jobId ? null : conversation?.result) ?? null);
  const paper = tab.paperId ? collection.paperById.get(tab.paperId) : undefined;
  const summary = collection.summaries.find(item => item.paper_id === tab.paperId);
  const artifacts = job?.artifacts ?? [];
  const markdownArtifact = artifacts.find(item => item.kind === 'markdown' && item.paper_id === tab.paperId)
    ?? (paper?.markdown_artifact_id
      ? { artifact_id: paper.markdown_artifact_id, name: `${paper.title}.md`, kind: 'markdown', paper_id: paper.paper_id } satisfies Artifact
      : undefined);
  const registeredPdf = artifacts.find(item => item.kind === 'pdf' && item.paper_id === tab.paperId);
  const fallbackPdf = paper?.pdf_artifact_id
    ? { artifact_id: paper.pdf_artifact_id, name: `${paper.title}.pdf`, kind: 'pdf', paper_id: paper.paper_id } satisfies Artifact
    : undefined;
  const defaultPdfArtifact = registeredPdf ?? fallbackPdf;
  const [view, setView] = useState<ReaderView>('summary');
  const [fulltext, setFulltext] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    setView('summary');
    setFulltext(null);
    setTruncated(false);
    setLocalError(null);
  }, [tab.paperId, tab.jobId]);

  useEffect(() => {
    if (view !== 'markdown' || !job || !markdownArtifact) return;
    let current = true;
    setLoading(true);
    setLocalError(null);
    void Promise.resolve().then(() => desktopApi().readArtifact(job.job_id, markdownArtifact.artifact_id)).then(value => {
      if (!current) return;
      setFulltext(value.text);
      setTruncated(value.truncated);
    }).catch(error => {
      if (current) setLocalError(error instanceof Error ? error.message : String(error));
    }).finally(() => {
      if (current) setLoading(false);
    });
    return () => { current = false; };
  }, [view, job?.job_id, markdownArtifact?.artifact_id]);

  if ((loadingConversation || remoteLoading) && !paper) {
    return <div className="dock-panel-skeleton" aria-live="polite">
      <span className="skeleton-line wide"/><span className="skeleton-line"/><span className="skeleton-line medium"/>
    </div>;
  }
  if (remoteError?.jobId === remoteTargetId) return <div className="panel-empty"><span><FileText size={20}/></span><strong>所属任务暂时不可读取</strong><p>{remoteError.message}</p></div>;
  if (!paper) return <div className="panel-empty"><span><FileText size={20}/></span><strong>该论文不在所属任务结果中</strong><p>论文标签会保留最近打开的任务上下文；可在标签栏关闭后从当前结果重新打开。</p></div>;

  const citationPapers = [...collection.paperById.values(), ...(job?.result?.report?.references ?? [])];
  const readingText = paperTitleCitations(summary?.deep_analysis?.trim() || summary?.markdown || (paper.abstract ? `## 摘要\n\n${paper.abstract}` : ''), citationPapers);
  const sourceUrl = safeHttpUrl(paper.paper_url);
  const derivedSummaryStatus = summaryStatus(summary);
  const analysisIds = collection.result?.metrics?.analysis_paper_ids;
  const analyzed = Array.isArray(analysisIds) && analysisIds.includes(paper.paper_id);
  const sourceNote = summary?.summary_scope === 'demo_fixture' ? '演示样例'
    : summary?.summary_scope === 'abstract_only' ? '基于论文摘要的研究笔记' : '';
  const summaryOverview = paperTitleCitations(summary
    ? (summary.overview?.trim() || summary.core_idea?.trim() || summary.relevance?.trim() || '')
    : '', citationPapers);

  const copy = async () => {
    if (!readingText) return;
    try {
      await desktopApi().copyText(readingText);
      workspaceStore.setNotice('研究笔记已复制。');
    } catch (error) {
      workspaceStore.setError(error instanceof Error ? error.message : String(error));
    }
  };
  const openArtifact = async (artifact: Artifact) => {
    if (!job) return;
    try {
      await desktopApi().openArtifact(job.job_id, artifact.artifact_id);
    } catch (error) {
      workspaceStore.setError(error instanceof Error ? error.message : String(error));
    }
  };
  const exportArtifact = async (artifact: Artifact) => {
    if (!job) return;
    try {
      const response = await desktopApi().exportArtifact(job.job_id, artifact.artifact_id);
      workspaceStore.setNotice(response.cancelled ? '已取消下载。' : response.path ? `已下载到：${response.path}` : '下载完成。');
    } catch (error) {
      workspaceStore.setError(error instanceof Error ? error.message : String(error));
    }
  };
  const openSource = async () => {
    if (!sourceUrl) return;
    try {
      await desktopApi().openExternal(sourceUrl);
    } catch (error) {
      workspaceStore.setError(error instanceof Error ? error.message : String(error));
    }
  };
  return <article className="reader-view">
    <header className="reader-header">
      <div className="reader-icon"><BookOpen size={17}/></div>
      <div className="reader-title">
        {sourceNote && <span>{sourceNote}</span>}
        <h2 title={paper.title}>{paper.title}</h2>
        <p>{paper.authors.join(', ')}{paper.year ? ` · ${paper.year}` : ''}</p>
      </div>
      <details className="reader-more">
        <summary className="icon-button" aria-label="更多论文操作" title="更多操作"><MoreHorizontal size={17}/></summary>
        <div className="reader-more-menu">
          {defaultPdfArtifact && <button type="button" onClick={() => void openArtifact(defaultPdfArtifact)}><FolderOpen size={14}/>在外部打开 PDF</button>}
          {defaultPdfArtifact && <button type="button" onClick={() => void exportArtifact(defaultPdfArtifact)}><Download size={14}/>下载 PDF</button>}
          {sourceUrl && <button type="button" onClick={() => void openSource()}><ArrowUpRight size={14}/>打开论文来源页面</button>}
        </div>
      </details>
    </header>

    <div className="reader-switch" role="tablist" aria-label={`${paper.title} 阅读视图`}>
      <button type="button" role="tab" aria-selected={view === 'summary'} className={view === 'summary' ? 'is-active' : ''} onClick={() => setView('summary')}>研究笔记</button>
      <button type="button" role="tab" aria-selected={view === 'markdown'} className={view === 'markdown' ? 'is-active' : ''} disabled={!markdownArtifact} onClick={() => setView('markdown')}>原文 Markdown</button>
      <button type="button" role="tab" aria-selected={view === 'pdf'} className={view === 'pdf' ? 'is-active' : ''} disabled={!defaultPdfArtifact} onClick={() => setView('pdf')}>PDF 原文</button>
    </div>

    {view === 'summary' && <div role="tabpanel" aria-label="研究笔记">
      {summaryOverview && <section className="summary-overview" aria-label="研究笔记概览"><span>研究笔记概览</span><p>{summaryOverview}</p></section>}
      {paper.selection_reason ? <div className="scope-note"><span>精选依据</span><strong>{paper.selection_reason}</strong></div> : null}
      {(derivedSummaryStatus === 'failed' || derivedSummaryStatus === 'missing') && analyzed && <div className="inline-warning" role="status">
        {derivedSummaryStatus === 'failed'
          ? '研究笔记生成失败。'
          : '暂无研究笔记。'}
      </div>}
      <div className="reader-actions"><button className="subtle-button" type="button" disabled={!readingText} onClick={() => void copy()}><Copy size={14}/>复制研究笔记</button></div>
      {readingText ? <MarkdownContent text={readingText}/> : <div className="panel-empty compact-empty"><span><FileText size={19}/></span><strong>没有可读研究笔记</strong><p>当前结果未提供这篇论文的摘要或研究笔记。</p></div>}
    </div>}
    {view === 'pdf' && (job && defaultPdfArtifact ? <PdfViewer
      key={defaultPdfArtifact.artifact_id}
      jobId={job.job_id}
      artifact={defaultPdfArtifact}
      initialPage={1}
    /> : <div className="inline-warning">暂无 PDF 原文。</div>)}

    {view === 'markdown' && <div className="reader-markdown" role="tabpanel" aria-label="原文 Markdown">
      {loading && <div className="reader-loading" aria-live="polite"><span className="skeleton-line wide"/><span className="skeleton-line"/></div>}
      {fulltext && <MarkdownContent text={fulltext}/>}
      {truncated && <div className="inline-warning" role="status">全文超过 2 MiB，当前仅显示服务端返回的截断内容。</div>}
      {localError && <div className="inline-error" role="alert">Markdown 读取失败：{localError}</div>}
      {!loading && !localError && !fulltext && <p className="panel-note">暂无原文 Markdown。</p>}
    </div>}
  </article>;
}
