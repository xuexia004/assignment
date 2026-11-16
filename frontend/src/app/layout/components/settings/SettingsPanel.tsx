import { useEffect, useId, useState } from 'react';
import { Check, Eye, EyeOff, FolderOpen, LoaderCircle, LockKeyhole, Monitor, Moon, Save, ShieldCheck, Sun, Trash2 } from 'lucide-react';
import { type ConfigUpdate, type PublicConfig } from '../../../types/contracts';
import { desktopApi } from '../../../services/desktopApi';
import { useWorkspaceState, workspaceStore } from '../../../stores/workspaceStore';
import { layoutStore, useLayoutState } from '../../stores/layoutStore';

const themes = [
  { value: 'system', label: '跟随系统', icon: Monitor },
  { value: 'light', label: '浅色', icon: Sun },
  { value: 'dark', label: '深色', icon: Moon },
] as const;

type Methods = Required<Pick<PublicConfig, 'crossref_enabled' | 'cross_encoder_enabled' | 'pdf_parser' | 'paper_analysis_strategy'>>;
const BASELINE: Methods = { crossref_enabled: false, cross_encoder_enabled: false, pdf_parser: 'legacy', paper_analysis_strategy: 'direct' };
const DEFAULT_BUDGETS = { download_candidate_timeout: 45, download_timeout: 180, fulltext_timeout: 180 };
const SLOW_BUDGETS = { download_candidate_timeout: 90, download_timeout: 240, fulltext_timeout: 600 };
type BudgetPreset = 'default' | 'slow' | 'custom';
function budgetPresetFor(values: typeof DEFAULT_BUDGETS): BudgetPreset {
  const matches = (preset: typeof DEFAULT_BUDGETS) => values.download_candidate_timeout === preset.download_candidate_timeout
    && values.download_timeout === preset.download_timeout && values.fulltext_timeout === preset.fulltext_timeout;
  return matches(DEFAULT_BUDGETS) ? 'default' : matches(SLOW_BUDGETS) ? 'slow' : 'custom';
}
const ANALYSIS_LABELS = {
  grounded: '全文证据先行', evidence: '篇内混合检索', evidence_fast: '篇内级联检索', sections: '分段并发分析',
} as const;

export function SettingsPanel() {
  const { config, serviceInfo } = useWorkspaceState();
  const { theme } = useLayoutState();
  const themeId = useId();
  const [baseUrl, setBaseUrl] = useState('https://api.deepseek.com');
  const [model, setModel] = useState('deepseek-flash');
  const [proxy, setProxy] = useState('');
  const [topK, setTopK] = useState(5);
  const [concurrency, setConcurrency] = useState(3);
  const [llmTimeout, setLlmTimeout] = useState(180);
  const [httpTimeout, setHttpTimeout] = useState(30);
  const [thinking, setThinking] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [openAlexKey, setOpenAlexKey] = useState('');
  const [clearApiKey, setClearApiKey] = useState(false);
  const [clearOpenAlexKey, setClearOpenAlexKey] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [showOpenAlexKey, setShowOpenAlexKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [methods, setMethods] = useState<Methods>(BASELINE);
  const [analysisMethod, setAnalysisMethod] = useState<Exclude<Methods['paper_analysis_strategy'], 'direct'>>('grounded');
  const [reportStrategy, setReportStrategy] = useState<NonNullable<PublicConfig['report_strategy']>>('legacy');
  const reportStrategyAvailable = config?.report_strategy !== undefined;
  useEffect(() => {
    if (config?.report_strategy !== undefined) setReportStrategy(config.report_strategy);
  }, [config?.report_strategy]);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [budgets, setBudgets] = useState(DEFAULT_BUDGETS);
  const [budgetPreset, setBudgetPreset] = useState<BudgetPreset>('default');
  const [budgetError, setBudgetError] = useState<string | null>(null);
  const budgetsAvailable = config?.download_candidate_timeout !== undefined && config?.download_timeout !== undefined && config?.fulltext_timeout !== undefined;
  useEffect(() => {
    if (budgetsAvailable && config) {
      const values = { download_candidate_timeout: config.download_candidate_timeout!, download_timeout: config.download_timeout!, fulltext_timeout: config.fulltext_timeout! };
      setBudgets(values); setBudgetPreset(budgetPresetFor(values));
    }
  }, [config?.download_candidate_timeout, config?.download_timeout, config?.fulltext_timeout, budgetsAvailable]);
  const saveBudgets = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault(); setSaving(true); setBudgetError(null);
    try { workspaceStore.updateConfig(await desktopApi().saveConfig(budgets)); }
    catch (reason) { setBudgetError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setSaving(false); }
  };

  const methodsAvailable = config?.crossref_enabled !== undefined && config?.cross_encoder_enabled !== undefined
    && config?.pdf_parser !== undefined && config?.paper_analysis_strategy !== undefined;
  useEffect(() => {
    if (!methodsAvailable || !config) return;
    setMethods({ crossref_enabled: config.crossref_enabled!, cross_encoder_enabled: config.cross_encoder_enabled!,
      pdf_parser: config.pdf_parser!, paper_analysis_strategy: config.paper_analysis_strategy! });
    if (config.paper_analysis_strategy !== 'direct') setAnalysisMethod(config.paper_analysis_strategy!);
  }, [config?.crossref_enabled, config?.cross_encoder_enabled, config?.pdf_parser, config?.paper_analysis_strategy, methodsAvailable]);

  useEffect(() => {
    if (!config) return;
    setBaseUrl(config.base_url);
    setModel(config.model);
    setProxy(config.http_proxy);
    setTopK(config.default_top_k);
    setConcurrency(config.concurrency);
    setLlmTimeout(config.llm_timeout);
    setHttpTimeout(config.http_timeout);
    setThinking(config.thinking_enabled);
    setApiKey('');
    setOpenAlexKey('');
    setClearApiKey(false);
    setClearOpenAlexKey(false);
  }, [config?.base_url, config?.model, config?.http_proxy, config?.default_top_k, config?.concurrency,
    config?.llm_timeout, config?.http_timeout, config?.thinking_enabled, config?.api_key_configured,
    config?.openalex_api_key_configured]);

  const saveProfile = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setProfileError(null);
    try {
      workspaceStore.updateConfig(await desktopApi().saveConfig({ research_profile: 'custom', ...methods,
        ...(reportStrategyAvailable ? { report_strategy: reportStrategy } : {}) }));
    } catch (reason) {
      setProfileError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  };

  const save = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!baseUrl.trim() || !model.trim()) {
      setError('请填写 API Base URL 和模型名称。');
      return;
    }
    setSaving(true);
    setError(null);
    const values: ConfigUpdate = {
      base_url: baseUrl.trim(),
      model: model.trim(),
      http_proxy: proxy.trim(),
      default_top_k: topK,
      concurrency,
      llm_timeout: llmTimeout,
      http_timeout: httpTimeout,
      thinking_enabled: thinking,
      ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
      ...(openAlexKey.trim() ? { openalex_api_key: openAlexKey.trim() } : {}),
      ...(clearApiKey ? { clear_api_key: true } : {}),
      ...(clearOpenAlexKey ? { clear_openalex_api_key: true } : {}),
    };
    try {
      const next: PublicConfig = await desktopApi().saveConfig(values);
      workspaceStore.updateConfig(next);
      setApiKey('');
      setOpenAlexKey('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  };

  const openDataDirectory = async () => {
    try {
      await desktopApi().openDataDirectory();
    } catch (reason) {
      workspaceStore.setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  if (!config) return <div className="panel-empty"><span><LockKeyhole size={20}/></span><strong>配置暂不可用</strong><p>桌面服务加载配置失败。请查看页面错误提示后重试。</p></div>;

  return <div className="settings-view">
    <header className="panel-intro"><div><span className="eyebrow">本机配置</span><h2>设置</h2></div><span className="settings-safe"><ShieldCheck size={14}/>密钥不回显</span></header>
    <p className="panel-subtitle">主题立即生效；连接参数修改后需点击保存。</p>
    <section className="settings-section" aria-label="外观">
      <div className="settings-section-head"><div><h3>外观</h3><p>选择界面主题，自动保存。</p></div></div>
      <div className="theme-options" role="radiogroup" aria-label="界面主题">
        {themes.map(({ value, label, icon: Icon }) => <label key={value} className="theme-choice">
          <input type="radio" name={themeId} value={value} checked={theme === value} onChange={() => layoutStore.setTheme(value)}/>
          <Icon size={17} aria-hidden="true"/><span>{label}</span>
        </label>)}
      </div>
    </section>
    <section className="settings-section" aria-label="研究方法开关">
      <div className="settings-section-head"><div><h3>研究方法开关</h3><p>逐项选择改进，关闭即使用对应基线。保存后用于新任务。</p></div></div>
      <form className="settings-form" onSubmit={event => void saveProfile(event)}>
        {!methodsAvailable && <p className="inline-error" role="alert">当前后台尚不支持单项开关，请在已有任务结束后重启桌面服务。</p>}
        <fieldset className="research-methods" disabled={saving || !methodsAvailable}>
          <legend className="sr-only">独立技术开关</legend>
          <label className="research-method-row"><span><small className="research-stage">01 · 论文来源</small><strong>新增 Crossref 数据源</strong><small>关闭：arXiv + OpenAlex。开启：额外查询期刊与会议题录。</small></span>
            <input className="research-switch" type="checkbox" role="switch" aria-label="新增 Crossref 数据源" checked={methods.crossref_enabled} onChange={event => setMethods({ ...methods, crossref_enabled: event.target.checked })}/></label>
          <label className="research-method-row"><span><small className="research-stage">02 · 候选排序</small><strong>联合重排</strong><small>关闭：语义融合排序。开启：对前 20 篇候选联合评分。</small></span>
            <input className="research-switch" type="checkbox" role="switch" aria-label="联合重排" checked={methods.cross_encoder_enabled} onChange={event => setMethods({ ...methods, cross_encoder_enabled: event.target.checked })}/></label>
          <label className="research-method-row"><span><small className="research-stage">03 · 全文转换</small><strong>PDF 版面解析</strong><small>关闭：原文本解析。开启：识别阅读顺序、章节与表格。</small></span>
            <input className="research-switch" type="checkbox" role="switch" aria-label="PDF 版面解析" checked={methods.pdf_parser === 'layout'} onChange={event => setMethods({ ...methods, pdf_parser: event.target.checked ? 'layout' : 'legacy' })}/></label>
          <label className="research-method-row"><span><small className="research-stage">04 · 单篇分析</small><strong>单篇分析增强</strong><small>关闭：整篇直接分析。开启：使用下方所选方法。</small></span>
            <input className="research-switch" type="checkbox" role="switch" aria-label="单篇分析增强" checked={methods.paper_analysis_strategy !== 'direct'} onChange={event => setMethods({ ...methods, paper_analysis_strategy: event.target.checked ? analysisMethod : 'direct' })}/></label>
          {methods.paper_analysis_strategy !== 'direct' && <div className="research-analysis-choice">
            <label className="field"><span>分析方法</span><select aria-label="分析方法" value={analysisMethod} onChange={event => {
              const value = event.target.value as typeof analysisMethod;
              setAnalysisMethod(value); setMethods({ ...methods, paper_analysis_strategy: value });
            }}><optgroup label="完整原文"><option value="grounded">{ANALYSIS_LABELS.grounded}</option></optgroup>
              <optgroup label="篇内检索：两种替代方案"><option value="evidence">{ANALYSIS_LABELS.evidence}</option><option value="evidence_fast">{ANALYSIS_LABELS.evidence_fast}</option></optgroup>
              <optgroup label="实验性篇内并发"><option value="sections">{ANALYSIS_LABELS.sections}</option></optgroup></select></label>
            <p className="panel-note">{analysisMethod === 'grounded' ? '保留全部原文，先提取引句再回答；准确性提升仍待验证。' : analysisMethod === 'sections' ? '长文分段后并发分析；已有小样本更慢，保留作实验对照。' : analysisMethod === 'evidence' ? '混合检索选取相关段落，减少输入；属于部分原文分析。' : '对照方法是篇内混合检索：改用 BM25 初筛再重排；新文档准备更快，回答质量有取舍。'}</p>
          </div>}
          <div className="research-analysis-choice">
            <small className="research-stage">05 · 综合报告</small>
            <label className="field"><span>综合报告方法</span><select aria-label="综合报告方法" value={reportStrategyAvailable ? reportStrategy : ''} disabled={!reportStrategyAvailable}
              onChange={event => setReportStrategy(event.target.value as typeof reportStrategy)}>
              {!reportStrategyAvailable && <option value="">当前后台尚不支持</option>}
              <option value="legacy">旧版自由生成（基线）</option>
              <option value="structured">结构化生成与格式修复</option>
            </select></label>
            <p className="panel-note">{reportStrategyAvailable ? '两种方案均保留全部输入笔记。结构化方案分项生成后排版，格式错误最多修复一次；与单篇分析方法独立。' : '重启桌面服务后可选择综合报告方法；当前仍可保存其他研究开关。'}</p>
          </div>
        </fieldset>
        <p className="panel-note">不同环节可以组合；单篇分析方案每次只选一种。多篇论文的并行优化仍在对话输入区切换。</p>
        {profileError && <div className="inline-error" role="alert">{profileError}</div>}
        <div className="research-method-actions">
          <button className="secondary-button" type="submit" disabled={saving || !methodsAvailable}>{saving ? <LoaderCircle size={14} className="spin"/> : <Check size={14}/>}保存研究方法</button>
          <button className="secondary-button" type="button" disabled={saving || !methodsAvailable} onClick={() => { setMethods({ ...BASELINE }); setReportStrategy('legacy'); }}>全部恢复基线</button>
        </div>
      </form>
    </section>
    <section className="settings-section">
      <div className="settings-section-head"><div><h3>DeepSeek</h3><p>研究与讨论使用的模型服务。</p></div><span className={`configured-pill ${config.api_key_configured ? 'is-ready' : ''}`}><i/>{config.api_key_configured ? '已配置' : '待配置'}</span></div>
      <form className="settings-form" onSubmit={event => void save(event)}>
        <label className="field"><span>API Base URL</span><input value={baseUrl} onChange={event => setBaseUrl(event.target.value)} autoComplete="url"/></label>
        <label className="field"><span>模型名称</span><input value={model} onChange={event => setModel(event.target.value)} placeholder="例如 deepseek-flash"/></label>
        <label className="field"><span>API Key <small>{config.api_key_configured ? '已保存，留空则保留' : '尚未配置'}</small></span><span className="secret-field"><input type={showApiKey ? 'text' : 'password'} value={apiKey} onChange={event => { setApiKey(event.target.value); if (event.target.value) setClearApiKey(false); }} autoComplete="new-password" placeholder="输入新密钥"/><button type="button" className="field-icon" aria-label={showApiKey ? '隐藏密钥' : '显示密钥'} onClick={() => setShowApiKey(value => !value)}>{showApiKey ? <EyeOff size={15}/> : <Eye size={15}/>}</button></span></label>
        <label className="clear-key"><input type="checkbox" checked={clearApiKey} onChange={event => { setClearApiKey(event.target.checked); if (event.target.checked) setApiKey(''); }}/><Trash2 size={14}/>明确清除已保存的 DeepSeek Key</label>
        <label className="field"><span>HTTP Proxy</span><input value={proxy} onChange={event => setProxy(event.target.value)} placeholder="留空表示不使用代理" autoComplete="url"/></label>
        <label className="toggle-field"><span><strong>思考模式</strong><small>按服务端模型配置启用。</small></span><input type="checkbox" checked={thinking} onChange={event => setThinking(event.target.checked)}/></label>
        <div className="settings-grid">
          <label className="field"><span>默认精选论文</span><select aria-label="默认精选论文数量" value={topK} onChange={event => setTopK(Number(event.target.value))}>{![3, 5, 8].includes(topK) && <option value={topK}>其他 {topK} 篇</option>}<option value={3}>3 篇</option><option value={5}>5 篇</option><option value={8}>8 篇</option></select></label>
          <label className="field"><span>并发数</span><input type="number" min={1} max={8} value={concurrency} onChange={event => setConcurrency(Number(event.target.value))}/></label>
          <label className="field"><span>模型超时（秒）</span><input type="number" min={5} max={600} value={llmTimeout} onChange={event => setLlmTimeout(Number(event.target.value))}/></label>
          <label className="field"><span>HTTP 超时（秒）</span><input type="number" min={5} max={300} value={httpTimeout} onChange={event => setHttpTimeout(Number(event.target.value))}/></label>
        </div>
        {error && <div className="inline-error" role="alert">{error}</div>}
        <button className="primary-button save-settings" type="submit" disabled={saving}>{saving ? <LoaderCircle size={15} className="spin"/> : <Save size={15}/>}保存 DeepSeek 设置</button>
      </form>
    </section>
    <section className="settings-section">
      <div className="settings-section-head"><div><h3>OpenAlex</h3><p>用于学术元数据检索的独立凭据。</p></div><span className={`configured-pill ${config.openalex_api_key_configured ? 'is-ready' : ''}`}><i/>{config.openalex_api_key_configured ? '已配置' : '待配置'}</span></div>
      <form className="settings-form" onSubmit={event => void save(event)}>
        <label className="field"><span>OpenAlex API Key <small>{config.openalex_api_key_configured ? '已保存，留空则保留' : '尚未配置'}</small></span><span className="secret-field"><input type={showOpenAlexKey ? 'text' : 'password'} value={openAlexKey} onChange={event => { setOpenAlexKey(event.target.value); if (event.target.value) setClearOpenAlexKey(false); }} autoComplete="new-password" placeholder="输入新密钥"/><button type="button" className="field-icon" aria-label={showOpenAlexKey ? '隐藏密钥' : '显示密钥'} onClick={() => setShowOpenAlexKey(value => !value)}>{showOpenAlexKey ? <EyeOff size={15}/> : <Eye size={15}/>}</button></span></label>
        <label className="clear-key"><input type="checkbox" checked={clearOpenAlexKey} onChange={event => { setClearOpenAlexKey(event.target.checked); if (event.target.checked) setOpenAlexKey(''); }}/><Trash2 size={14}/>明确清除已保存的 OpenAlex Key</label>
        <button className="secondary-button" type="submit" disabled={saving}>{saving ? <LoaderCircle size={14} className="spin"/> : <Check size={14}/>}保存 OpenAlex 设置</button>
      </form>
    </section>
    <section className="settings-section" aria-label="下载与全文处理">
      <div className="settings-section-head"><div><h3>下载与全文处理</h3><p>调整等待上限，保存后用于新任务。</p></div></div>
      {!budgetsAvailable && <p className="inline-error">后台需要更新，请在任务结束后重启桌面服务。</p>}
      <form className="settings-form" onSubmit={event => void saveBudgets(event)}>
        <fieldset className="research-methods" disabled={saving || !budgetsAvailable}>
          <label className="field"><span>时间预算方案</span><select value={budgetPreset} onChange={event => {
            const preset = event.target.value as BudgetPreset;
            setBudgetPreset(preset);
            if (preset !== 'custom') setBudgets({ ...(preset === 'slow' ? SLOW_BUDGETS : DEFAULT_BUDGETS) });
          }}><option value="default">默认预算</option><option value="slow">慢速网络</option><option value="custom">自定义</option></select></label>
          <label className="field"><span>单个 PDF 地址限时（秒）</span><input type="number" min={5} max={600} required value={budgets.download_candidate_timeout} onChange={event => { setBudgetPreset('custom'); setBudgets({ ...budgets, download_candidate_timeout: Number(event.target.value) }); }}/></label>
          <label className="field"><span>单篇下载总限时（秒）</span><input type="number" min={5} max={1800} required value={budgets.download_timeout} onChange={event => { setBudgetPreset('custom'); setBudgets({ ...budgets, download_timeout: Number(event.target.value) }); }}/></label>
          <label className="field"><span>整个全文阶段限时（秒）</span><input type="number" min={5} max={1200} required value={budgets.fulltext_timeout} onChange={event => { setBudgetPreset('custom'); setBudgets({ ...budgets, fulltext_timeout: Number(event.target.value) }); }}/></label>
          <p className="panel-note">可选择预设，也可直接修改数值，修改后自动切换为自定义。点击“保存时间预算”后生效。</p>
          <p className="panel-note">默认：45 / 180 / 180 秒；慢速网络：90 / 240 / 600 秒。全文阶段包含下载、解析和失败替补。上限越大，慢速或失败任务等待越久，完成后会提前结束。</p>
          {(budgets.download_timeout < budgets.download_candidate_timeout || budgets.fulltext_timeout < budgets.download_timeout) && <p className="panel-note">外层总限时较短，会提前结束内部下载。</p>}
          {budgetError && <p className="inline-error" role="alert">{budgetError}</p>}
          <button className="secondary-button" type="submit">保存时间预算</button>
        </fieldset>
      </form>
    </section>
    <section className="settings-section settings-service">
      <div className="settings-section-head"><div><h3>桌面服务</h3><p>本机服务状态与研究数据位置。</p></div><button className="subtle-button" type="button" onClick={() => void openDataDirectory()}><FolderOpen size={14}/>打开数据目录</button></div>
      {serviceInfo ? <dl className="service-facts"><div><dt>版本</dt><dd>{serviceInfo.version} · API v{serviceInfo.api_version}</dd></div><div><dt>运行中任务</dt><dd>{serviceInfo.active_jobs}</dd></div><div><dt>数据目录</dt><dd className="service-path"><details><summary title={serviceInfo.data_root} aria-label="展开或收起完整数据目录">{serviceInfo.data_root}</summary><div className="service-path-full">{serviceInfo.data_root}</div></details></dd></div></dl> : <p className="panel-note">服务信息暂不可用。</p>}
    </section>
  </div>;
}
