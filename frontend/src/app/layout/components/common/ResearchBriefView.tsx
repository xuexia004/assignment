import { ArrowRight, BookOpen, FileText, Lightbulb } from 'lucide-react';
import type { Job, KeyPaper, Paper, PipelineResult } from '../../../types/contracts';
import { layoutStore } from '../../stores/layoutStore';
import {
  deliveryStatusFor,
  reportArtifactFor,
  supportedFindings,
} from '../../../utils/researchBrief';
import { paperTitleCitations } from '../../../utils/paperCitations';

export function ResearchBriefView({ result, job }: { result: PipelineResult; job?: Job | null }) {
  const brief = result.brief;
  const delivery = deliveryStatusFor(result, job);
  if (!brief && !delivery) return null;
  const findings = supportedFindings(brief);
  const papers = new Map<string, Paper>();
  for (const paper of [...result.selected_papers, ...result.papers]) papers.set(paper.paper_id, paper);
  const keyPapers = (brief?.key_papers ?? [])
    .map(item => ({ item, paper: papers.get(item.paper_id) }))
    .filter((value): value is { item: KeyPaper; paper: Paper } => Boolean(value.paper));
  const reportArtifact = reportArtifactFor(result, job?.artifacts ?? []);
  const hasLegacyReport = Boolean(result.report?.markdown?.trim());
  const citationPapers = [...papers.values(), ...(result.report?.references ?? [])];

  return <section className={`research-brief delivery-${delivery ?? 'legacy'}`} aria-label="研究简报">
    <header className="brief-header">
      <h2>研究简报</h2>
    </header>

    {findings.length > 0 && <ol className="brief-findings">
      {findings.map((finding, index) => <li key={`${index}:${finding.statement}`}>
          <span className="finding-number">{index + 1}</span>
          <div><p>{paperTitleCitations(finding.statement, citationPapers)}</p></div>
        </li>)}
    </ol>}

    {brief && findings.length === 0 && <p className="brief-empty">暂无研究结论，可以先阅读已获取的论文。</p>}

    {keyPapers.length > 0 && <section className="brief-key-papers" aria-label="关键论文">
      <h3><BookOpen size={14}/>关键论文 <span>{keyPapers.length}</span></h3>
      <div>{keyPapers.map(({ item, paper }) => <button type="button" key={item.paper_id} onClick={() => layoutStore.openPaper(paper.paper_id, paper.title, job?.job_id)}>
        <strong>{paper.title}</strong>
        {item.selection_reason && <p>{item.selection_reason}</p>}
        <ArrowRight size={14}/>
      </button>)}</div>
    </section>}

    {brief?.next_step?.action && <section className="brief-next-step" aria-label="推荐下一步">
      <span><Lightbulb size={16}/></span>
      <div><h3>推荐下一步</h3><p>{paperTitleCitations(brief.next_step.action, citationPapers)}</p>{brief.next_step.rationale && <small>{paperTitleCitations(brief.next_step.rationale, citationPapers)}</small>}</div>
    </section>}

    {(reportArtifact || hasLegacyReport) && <button className="brief-report-link" type="button" onClick={() => layoutStore.openPanel('report')}>
      <FileText size={16}/><span>阅读研究报告</span><ArrowRight size={15}/>
    </button>}

  </section>;
}
