import { Gauge } from 'lucide-react';
import type { Paper } from '../../../types/contracts';
import { useWorkspaceState } from '../../../stores/workspaceStore';
import { formatDuration } from '../../../utils/format';
import { stageLabel } from '../../../utils/labels';
import { metricNumber, taskRankings, taskTotalSeconds } from '../../../utils/taskMetrics';
import { layoutStore } from '../../stores/layoutStore';

const scoreLabel = (score: number | null) => score === null ? '未记录' : score.toFixed(3);

function RankingTable({ title, papers, mean, score, jobId }: {
  title: string; papers: Paper[]; mean: number | null;
  score: (paper: Paper) => number | null; jobId?: string;
}) {
  return <section className="task-ranking" aria-label={title}>
    <div className="task-metrics-heading"><h3>{title}</h3><span>均值 {scoreLabel(mean)}</span></div>
    {papers.length ? <table className="task-ranking-table">
      <thead><tr><th scope="col">排名</th><th scope="col">论文</th><th scope="col">相关性</th></tr></thead>
      <tbody>{papers.map((paper, index) => <tr key={paper.paper_id}>
        <td>{index + 1}</td>
        <td><button type="button" onClick={() => layoutStore.openPaper(paper.paper_id, paper.title, jobId)}>{paper.title}</button></td>
        <td>{scoreLabel(score(paper))}</td>
      </tr>)}</tbody>
    </table> : <p className="panel-note">暂无相关性记录。</p>}
  </section>;
}

export function MetricsPanel() {
  const { activeJob, resultJob, conversation } = useWorkspaceState();
  const job = resultJob ?? activeJob;
  const result = job?.result ?? conversation?.result;
  if (!result) return <div className="panel-empty"><span><Gauge size={20}/></span><strong>暂无性能数据</strong><p>完成研究后查看 Top-k 相关性与运行时间。</p></div>;
  const { semantic, selected, semanticMean, selectedMean } = taskRankings(result);
  const timings = (result.timings ?? []).filter(timing => timing.stage !== 'total');
  const ranking = result.metrics.ranking as { duration_seconds?: number; actual_ranker?: string } | undefined;
  const degraded = result.metrics.selector_degraded === true;

  return <div className="metrics-view">
    <header className="panel-intro"><h2>性能数据</h2></header>
    <p className="task-metrics-question">{result.idea}</p>
    {result.mode === 'demo' && <p className="task-metrics-note">演示数据</p>}
    <section className="task-runtime" aria-label="运行时间">
      <div className="task-metrics-heading"><h3>运行时间</h3><strong>{formatDuration(taskTotalSeconds(result) ?? undefined)}</strong></div>
      <dl><div><dt>执行方式</dt><dd>{result.execution_mode === 'concurrent' ? '并行' : result.execution_mode === 'sequential' ? '顺序' : '未记录'}</dd></div>{timings.map((timing, index) => <div key={`${timing.stage}:${index}`}>
        <dt>{stageLabel(timing.stage)}</dt><dd>{formatDuration(timing.duration_seconds ?? timing.wall_seconds)}</dd>
      </div>)}</dl>
      {metricNumber(ranking?.duration_seconds) !== null && ranking?.actual_ranker === 'semantic' && <p className="task-metrics-note">检索阶段中，语义排序用时 {formatDuration(ranking.duration_seconds)}。</p>}
    </section>
    <p className="task-metrics-note">语义检索使用余弦相似度；任务 Agent 使用筛选评分，两者分别展示。</p>
    <RankingTable title={`语义检索 Top-${result.top_k}`} papers={semantic} mean={semanticMean}
      score={paper => metricNumber(paper.ranking_details?.cosine_similarity)} jobId={job?.job_id}/>
    <RankingTable title={`任务 Agent 筛选 Top-${result.top_k}`} papers={selected} mean={degraded ? null : selectedMean}
      score={paper => degraded ? null : metricNumber(paper.final_score)} jobId={job?.job_id}/>
    {degraded && <p className="task-metrics-note">本次任务 Agent 筛选未完成，列表保留降级排序。</p>}
  </div>;
}
