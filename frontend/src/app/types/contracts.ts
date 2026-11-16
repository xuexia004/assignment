export type {
  ResearchProfile,
  Action,
  Artifact,
  ArtifactText,
  BenchmarkFile,
  ConfigUpdate,
  ContractError,
  Conversation,
  ConversationSummary,
  DeliveryStatus,
  DesktopAPI,
  EventItem,
  EvidenceLevel,
  EvidenceReference,
  ExecutionMode,
  Finding,
  Job,
  JobEvent,
  KeyPaper,
  Message,
  NextStep,
  Paper,
  PdfArtifact,
  PipelineResult,
  PreferredPaperCount,
  PublicConfig,
  RankingDetails,
  ReplacementAttempt,
  ReportReference,
  ResearchBrief,
  ResearchLimitation,
  ServiceInfo,
  Status,
  Summary,
  TargetCoverage,
  TargetCoverageEntry,
} from '../../../../shared/contracts';
export { RESEARCH_PROFILES, RESEARCH_PROFILE_LABELS } from '../../../../shared/contracts';

import type { Conversation } from '../../../../shared/contracts';

export function isReadOnlyConversation(conversation: Conversation | null | undefined): boolean {
  return (conversation as (Conversation & { read_only?: boolean }) | null | undefined)?.read_only === true;
}
