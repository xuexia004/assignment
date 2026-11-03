import type { Paper, PipelineResult } from '../types/contracts';

export function metricNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function taskRankings(result: PipelineResult) {
  const semantic = result.papers
    .filter(paper => metricNumber(paper.ranking_details?.cosine_similarity) !== null)
    .slice()
    .sort((a, b) => b.ranking_details!.cosine_similarity! - a.ranking_details!.cosine_similarity!)
    .slice(0, result.top_k);
  const selected = result.selected_papers.slice().sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity));
  const average = (papers: Paper[], score: (paper: Paper) => number | null) => {
    const values = papers.map(score).filter((value): value is number => value !== null);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  return {
    semantic,
    selected,
    semanticMean: average(semantic, paper => metricNumber(paper.ranking_details?.cosine_similarity)),
    selectedMean: average(selected, paper => metricNumber(paper.final_score)),
  };
}

export function taskTotalSeconds(result: PipelineResult): number | null {
  const total = result.timings?.find(timing => timing.stage === 'total');
  return metricNumber(result.metrics.main_agent_wall_seconds)
    ?? metricNumber(total?.duration_seconds) ?? metricNumber(total?.wall_seconds);
}
