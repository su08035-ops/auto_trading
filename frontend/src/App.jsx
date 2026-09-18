import { useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart3, Play, RefreshCw } from "lucide-react";
import { api, readJson, readText } from "./lib/api";
import { Badge, Button, Card, Input, Select, Spinner, cx } from "./components/ui";

const pct = (value) => `${((Number(value) || 0) * 100).toFixed(2)}%`;
const won = (value) => `${Math.round(Number(value) || 0).toLocaleString("ko-KR")}원`;
const num = (value) => Number(value || 0);
const sideLabel = (side) => (side === "buy" ? "매수" : "매도");
const sideTone = (side) => (side === "buy" ? "text-up" : "text-down");

function parseCsv(raw) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < raw.length; i += 1) {
    const char = raw[i];
    const next = raw[i + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const [header, ...body] = rows;
  return body.map((line) =>
    Object.fromEntries(header.map((name, index) => [name, line[index] ?? ""])),
  );
}

function toChartData(ohlcv, trades, equity) {
  const byDate = new Map(ohlcv.map((row) => [row.date, row]));
  const byTradeDate = new Map();
  for (const trade of trades) {
    const group = byTradeDate.get(trade.date) ?? { buy: null, sell: null, trades: [] };
    group[trade.trade_type] = trade.execution_price;
    group.trades.push(trade);
    byTradeDate.set(trade.date, group);
  }
  return equity
    .filter((row) => byDate.has(row.date))
    .map((row) => {
      const price = byDate.get(row.date);
      const trade = byTradeDate.get(row.date);
      return {
        date: row.date,
        close: price.close,
        buy: trade?.buy,
        sell: trade?.sell,
        trades: trade?.trades ?? [],
      };
    });
}

function moneyTick(value) {
  if (Math.abs(value) >= 100_000_000) return `${Math.round(value / 100_000_000)}억`;
  if (Math.abs(value) >= 10_000) return `${Math.round(value / 10_000)}만`;
  return Math.round(value).toLocaleString("ko-KR");
}

function TradeTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded border border-line bg-surface px-3 py-2 text-[12px] shadow-2xl">
      <div className="num text-muted mb-1.5">{label}</div>
      <div className="flex justify-between gap-8">
        <span className="text-muted">종가</span>
        <span className="num">{won(row.close)}</span>
      </div>
      {row.trades.map((trade) => (
        <div key={trade.id} className="flex justify-between gap-8">
          <span className={cx("font-medium", sideTone(trade.trade_type))}>
            {sideLabel(trade.trade_type)} {trade.shares_traded}주
          </span>
          <span className="num">{won(trade.execution_price)}</span>
        </div>
      ))}
    </div>
  );
}

function PriceTradeChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 10, right: 18, left: 10, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={(value) => value.slice(5)}
          minTickGap={42}
          tick={{ fill: "var(--muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={["dataMin", "dataMax"]}
          tickFormatter={moneyTick}
          tick={{ fill: "var(--muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={62}
        />
        <Tooltip content={<TradeTooltip />} />
        <Line
          type="monotone"
          dataKey="close"
          name="종가"
          stroke="var(--brand)"
          strokeWidth={1.8}
          dot={false}
        />
        <Scatter dataKey="buy" name="매수" fill="var(--up)" shape="triangle" />
        <Scatter dataKey="sell" name="매도" fill="var(--down)" shape="diamond" />
      </LineChart>
    </ResponsiveContainer>
  );
}

function EquityCurve({ data }) {
  const rows = data.map((row) => ({
    date: row.date,
    agent: row.agent,
    cash: row.cash,
    buyHold: row.buy_and_hold,
  }));
  return (
    <ResponsiveContainer width="100%" height={284}>
      <AreaChart data={rows} margin={{ top: 8, right: 18, left: 10, bottom: 0 }}>
        <defs>
          <linearGradient id="agentFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--brand)" stopOpacity={0.22} />
            <stop offset="100%" stopColor="var(--brand)" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={(value) => value.slice(5)}
          minTickGap={42}
          tick={{ fill: "var(--muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tickFormatter={moneyTick}
          tick={{ fill: "var(--muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={62}
        />
        <Tooltip
          content={({ active, payload, label }) =>
            active && payload?.length ? (
              <div className="rounded border border-line bg-surface px-3 py-2 text-[12px] shadow-2xl">
                <div className="num text-muted mb-1.5">{label}</div>
                {payload.map((item) => (
                  <div key={item.dataKey} className="flex justify-between gap-8">
                    <span style={{ color: item.color }}>{item.name}</span>
                    <span className="num">{won(item.value)}</span>
                  </div>
                ))}
              </div>
            ) : null
          }
        />
        <ReferenceLine y={10_000_000} stroke="var(--line)" strokeDasharray="4 4" />
        <Area
          type="monotone"
          dataKey="agent"
          name="에이전트"
          stroke="var(--brand)"
          strokeWidth={1.8}
          fill="url(#agentFill)"
          dot={false}
        />
        <Line type="monotone" dataKey="cash" name="현금" stroke="var(--muted)" dot={false} />
        <Line
          type="monotone"
          dataKey="buyHold"
          name="Buy & Hold"
          stroke="var(--indigo)"
          strokeWidth={1.4}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function MetricCard({ label, value, sub, tone }) {
  return (
    <Card className="min-h-[118px]">
      <div className="text-[12.5px] text-muted mb-3">{label}</div>
      <div className={cx("num text-[25px] leading-none", tone)}>{value}</div>
      {sub && <p className="mt-3 text-[12px] text-muted leading-relaxed">{sub}</p>}
    </Card>
  );
}

function FieldInput({ label, value, onChange, type = "text" }) {
  return (
    <label className="block">
      <span className="block text-[12.5px] text-muted mb-1.5">{label}</span>
      <Input
        type={type}
        value={value}
        onChange={(e) => onChange(type === "number" ? Number(e.target.value) : e.target.value)}
        mono={type !== "text"}
      />
    </label>
  );
}

function FieldSelect({ label, value, onChange, children }) {
  return (
    <label className="block">
      <span className="block text-[12.5px] text-muted mb-1.5">{label}</span>
      <Select value={value} onChange={(e) => onChange(e.target.value)}>
        {children}
      </Select>
    </label>
  );
}

export default function App() {
  const [config, setConfig] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [runData, setRunData] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const [draft, setDraft] = useState(null);
  const pollRef = useRef(null);

  const loadRuns = async (preferredId) => {
    const data = await api.runs();
    setRuns(data.runs ?? []);
    const next = preferredId || data.runs?.[0]?.id || "";
    if (next) setSelectedRunId(next);
    return data.runs ?? [];
  };

  useEffect(() => {
    let alive = true;
    Promise.all([api.config(), api.runs()])
      .then(([cfg, index]) => {
        if (!alive) return;
        setConfig(cfg);
        setDraft({
          source: cfg.data.source,
          algorithm: cfg.agent.algorithm,
          episodes: cfg.training.episodes,
          seed: cfg.training.seed,
          min_holding_days: cfg.environment.min_holding_days,
          start_date: cfg.data.start_date,
          validation_start: cfg.data.validation_start,
          split_date: cfg.data.split_date,
          end_date: cfg.data.end_date,
        });
        setRuns(index.runs ?? []);
        setSelectedRunId(index.runs?.[0]?.id ?? "");
      })
      .catch((e) => setError(e.message));
    return () => {
      alive = false;
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    const run = runs.find((item) => item.id === selectedRunId);
    if (!run) return;
    Promise.all([
      readText(run.paths.ohlcv),
      readText(run.paths.trades),
      readText(run.paths.equity),
      readJson(run.paths.metrics),
    ])
      .then(([ohlcvCsv, tradesCsv, equityCsv, metrics]) => {
        const ohlcv = parseCsv(ohlcvCsv).map((row) => ({
          date: row.date,
          open: num(row.open),
          high: num(row.high),
          low: num(row.low),
          close: num(row.close),
          volume: num(row.volume),
        }));
        const trades = parseCsv(tradesCsv)
          .filter((row) => Number(row.shares_traded) > 0)
          .map((row, index) => ({
            ...row,
            id: `${row.date}-${index}`,
            execution_price: num(row.execution_price),
            shares_traded: num(row.shares_traded),
            trade_value: num(row.trade_value),
            requested_target: num(row.requested_target),
            executed_target: num(row.executed_target),
          }));
        const equity = parseCsv(equityCsv).map((row) => ({
          date: row.date,
          agent: num(row.agent),
          cash: num(row.cash),
          buy_and_hold: num(row.buy_and_hold),
        }));
        setRunData({ run, ohlcv, trades, equity, metrics });
      })
      .catch((e) => setError(e.message));
  }, [runs, selectedRunId]);

  const chartData = useMemo(
    () => (runData ? toChartData(runData.ohlcv, runData.trades, runData.equity) : []),
    [runData],
  );

  const visibleTrades = useMemo(() => {
    if (!runData) return [];
    return runData.trades.filter((trade) => filter === "all" || trade.trade_type === filter);
  }, [runData, filter]);

  const startExperiment = async () => {
    setError("");
    setJob({ status: "queued", phase: "대기 중", progress: 0, logs: [] });
    try {
      const { job_id: jobId } = await api.startExperiment(draft);
      const tick = async () => {
        const next = await api.job(jobId);
        setJob(next);
        if (next.status === "done") {
          clearInterval(pollRef.current);
          pollRef.current = null;
          const nextRuns = await loadRuns(next.run_id);
          if (nextRuns.some((run) => run.id === next.run_id)) {
            setSelectedRunId(next.run_id);
          }
        }
        if (next.status === "error") {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      };
      await tick();
      pollRef.current = setInterval(() => tick().catch((e) => setError(e.message)), 1200);
    } catch (e) {
      setError(e.message);
      setJob(null);
    }
  };

  if (error && !config) {
    return (
      <div className="min-h-screen grid place-items-center bg-ink px-5">
        <Card title="연결 오류">
          <p className="text-up text-[13px]">{error}</p>
        </Card>
      </div>
    );
  }

  if (!draft) {
    return (
      <div className="min-h-screen grid place-items-center bg-ink">
        <Spinner label="실험 환경 불러오는 중" />
      </div>
    );
  }

  const agent = runData?.metrics.agent ?? {};
  const buyHold = runData?.metrics.buy_and_hold ?? {};
  const buys = runData?.trades.filter((trade) => trade.trade_type === "buy").length ?? 0;
  const sells = runData?.trades.filter((trade) => trade.trade_type === "sell").length ?? 0;
  const running = job?.status === "queued" || job?.status === "running";
  const progress = Math.round((job?.progress ?? 0) * 100);

  return (
    <div className="min-h-full bg-ink text-body">
      <div className="border-b border-line bg-surface">
        <div className="max-w-[1480px] mx-auto px-5 md:px-7 h-[62px] flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="w-[30px] h-[30px] rounded-sm bg-brand/12 border border-brand/30 grid place-items-center">
              <BarChart3 size={16} className="text-brand" />
            </span>
            <div>
              <h1 className="text-[15px] font-bold tracking-[0.08em]">KAIRO LAB</h1>
              <p className="text-[11.5px] text-muted">DQN 실험 콘솔</p>
            </div>
          </div>
          <Badge tone={running ? "brand" : "neutral"}>{running ? "학습 중" : "대기"}</Badge>
        </div>
      </div>

      <main className="max-w-[1480px] mx-auto px-5 md:px-7 py-6 space-y-4">
        <div className="grid grid-cols-1 xl:grid-cols-[430px_1fr] gap-4 items-start">
          <Card
            eyebrow="EXPERIMENT"
            title="실험 실행"
            action={
              <Button variant="primary" onClick={startExperiment} loading={running}>
                <Play size={14} /> 시작
              </Button>
            }
          >
            <div className="grid grid-cols-2 gap-3">
              <FieldSelect label="데이터" value={draft.source} onChange={(source) => setDraft({ ...draft, source })}>
                <option value="kis">KIS 삼성전자</option>
                <option value="synthetic">합성 데이터</option>
              </FieldSelect>
              <FieldSelect label="알고리즘" value={draft.algorithm} onChange={(algorithm) => setDraft({ ...draft, algorithm })}>
                <option value="dqn">DQN</option>
                <option value="double_dqn">Double DQN</option>
              </FieldSelect>
              <FieldInput label="에피소드" type="number" value={draft.episodes} onChange={(episodes) => setDraft({ ...draft, episodes })} />
              <FieldInput label="시드" type="number" value={draft.seed} onChange={(seed) => setDraft({ ...draft, seed })} />
              <FieldInput label="최소보유일" type="number" value={draft.min_holding_days} onChange={(min_holding_days) => setDraft({ ...draft, min_holding_days })} />
              <FieldInput label="학습 시작" type="date" value={draft.start_date} onChange={(start_date) => setDraft({ ...draft, start_date })} />
              <FieldInput label="검증 시작" type="date" value={draft.validation_start} onChange={(validation_start) => setDraft({ ...draft, validation_start })} />
              <FieldInput label="테스트 시작" type="date" value={draft.split_date} onChange={(split_date) => setDraft({ ...draft, split_date })} />
              <div className="col-span-2">
                <FieldInput label="종료일" type="date" value={draft.end_date} onChange={(end_date) => setDraft({ ...draft, end_date })} />
              </div>
            </div>

            {job && (
              <div className="mt-4 rounded border border-line bg-ink p-3">
                <div className="flex items-center justify-between gap-3 text-[12.5px]">
                  <span className="text-brand font-medium">{job.error || job.phase}</span>
                  <span className="num text-muted">{progress}%</span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-raise overflow-hidden">
                  <div className="h-full bg-brand" style={{ width: `${progress}%` }} />
                </div>
                <pre className="mt-3 max-h-[180px] overflow-auto text-[11.5px] leading-relaxed text-muted whitespace-pre-wrap">
                  {(job.logs ?? []).join("\n")}
                </pre>
              </div>
            )}
          </Card>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              label="에이전트 수익률"
              value={pct(agent.cumulative_return)}
              sub={`Buy & Hold ${pct(buyHold.cumulative_return)}`}
              tone={agent.cumulative_return >= 0 ? "text-up" : "text-down"}
            />
            <MetricCard
              label="최대낙폭"
              value={pct(agent.mdd)}
              sub={`Buy & Hold ${pct(buyHold.mdd)}`}
              tone="text-down"
            />
            <MetricCard
              label="거래 횟수"
              value={`${agent.trade_count ?? 0}회`}
              sub={`매수 ${buys}회 · 매도 ${sells}회`}
            />
            <MetricCard
              label="회전율"
              value={(agent.turnover ?? 0).toFixed(3)}
              sub={`평균 보유일 ${agent.average_holding_days ? agent.average_holding_days.toFixed(2) : "-"}`}
            />
          </div>
        </div>

        {error && (
          <Card>
            <p className="text-up text-[13px]">{error}</p>
          </Card>
        )}

        <Card
          eyebrow="RESULTS"
          title="결과 선택"
          action={
            <Button size="sm" variant="ghost" onClick={() => loadRuns(selectedRunId)}>
              <RefreshCw size={13} /> 새로고침
            </Button>
          }
        >
          <div className="flex flex-col md:flex-row gap-3 md:items-center">
            <div className="flex-1">
              <Select value={selectedRunId} onChange={(e) => setSelectedRunId(e.target.value)}>
                {runs.map((run) => (
                  <option key={run.id} value={run.id}>
                    {run.label}
                  </option>
                ))}
              </Select>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge tone="brand">{runData?.run.source ?? "-"}</Badge>
              <Badge>{runData?.run.algorithm ?? "-"}</Badge>
              <Badge>
                {runData?.run.period?.first_execution ?? "-"} ~ {runData?.run.period?.last_execution ?? "-"}
              </Badge>
            </div>
          </div>
        </Card>

        <div className="grid grid-cols-1 xl:grid-cols-[1.35fr_0.9fr] gap-4">
          <Card eyebrow="PRICE" title="가격과 매매 지점">
            {runData ? <PriceTradeChart data={chartData} /> : <Spinner />}
          </Card>

          <Card eyebrow="EQUITY" title="자산 곡선">
            {runData ? <EquityCurve data={runData.equity} /> : <Spinner />}
          </Card>
        </div>

        <Card
          eyebrow="TRADES"
          title="체결 내역"
          action={
            <div className="inline-flex rounded border border-line bg-ink p-1">
              {[
                ["all", "전체"],
                ["buy", "매수"],
                ["sell", "매도"],
              ].map(([value, label]) => (
                <button
                  key={value}
                  onClick={() => setFilter(value)}
                  className={cx(
                    "h-8 px-3 rounded-sm text-[12.5px] transition-colors",
                    filter === value ? "bg-raise text-body" : "text-muted hover:text-body",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          }
          pad={false}
        >
          <div className="overflow-auto max-h-[460px]">
            <table className="w-full text-[13px]">
              <thead className="sticky top-0 bg-surface z-10">
                <tr className="border-y border-line text-[11.5px] text-muted">
                  <th className="text-left font-medium px-5 py-3">체결일</th>
                  <th className="text-left font-medium px-5 py-3">판단일</th>
                  <th className="text-left font-medium px-5 py-3">구분</th>
                  <th className="text-right font-medium px-5 py-3">수량</th>
                  <th className="text-right font-medium px-5 py-3">체결가</th>
                  <th className="text-right font-medium px-5 py-3">거래대금</th>
                  <th className="text-right font-medium px-5 py-3">목표 비중</th>
                  <th className="text-right font-medium px-5 py-3">체결 후 비중</th>
                </tr>
              </thead>
              <tbody>
                {visibleTrades.map((trade) => (
                  <tr key={trade.id} className="border-b border-line hover:bg-raise/60">
                    <td className="px-5 py-3 num">{trade.date}</td>
                    <td className="px-5 py-3 num text-muted">{trade.decision_date}</td>
                    <td className={cx("px-5 py-3 font-medium", sideTone(trade.trade_type))}>
                      {sideLabel(trade.trade_type)}
                    </td>
                    <td className="px-5 py-3 text-right num">{trade.shares_traded.toLocaleString("ko-KR")}주</td>
                    <td className="px-5 py-3 text-right num">{won(trade.execution_price)}</td>
                    <td className="px-5 py-3 text-right num">{won(trade.trade_value)}</td>
                    <td className="px-5 py-3 text-right num">{pct(trade.requested_target)}</td>
                    <td className="px-5 py-3 text-right num">{pct(trade.executed_target)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </main>
    </div>
  );
}
