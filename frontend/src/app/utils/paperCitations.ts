export interface NamedPaper { paper_id: string; title: string; }

function escapePattern(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Present historical citations by title; machine IDs and stored artifacts stay intact. */
export function paperTitleCitations(text: string, papers: NamedPaper[]): string {
  const titles = new Map(papers.filter(paper => paper.title?.trim())
    .map(paper => [paper.paper_id, paper.title.replace(/\s+/g, ' ').trim()]));
  if (!text || !titles.size) return text;
  const titleText = (title: string) => title.replace(/([\\`*_{}\[\]<>#|])/g, '\\$1');
  const ids = [...titles.keys()].sort((a, b) => b.length - a.length).map(escapePattern).join('|');
  const idPattern = new RegExp(`(?<![\\w:/.-])(?:${ids})(?![\\w/.-])`, 'g');
  // Keep code and URLs byte-for-byte; only rewrite the reading text.
  return text.split(/(```[\s\S]*?```|`[^`\n]+`|https?:\/\/[^\s<>]+)/g).map((part, index) => {
    if (index % 2) return part;
    return part
      .replace(/^(\s*[-*]\s+)\[([^\]\n]+)\]\s+(.+)$/gm, (whole, prefix: string, id: string, title: string) =>
        titles.get(id) === title.trim() ? `${prefix}《${titleText(title)}》` : whole)
      .replace(/\[([^\]\n]+)\](?!\()/g, (whole, id: string) =>
        titles.has(id) ? `《${titleText(titles.get(id)!)}》` : whole)
      .replace(idPattern, id => titleText(titles.get(id)!));
  }).join('');
}
