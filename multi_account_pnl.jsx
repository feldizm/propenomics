import { useState, useCallback, useEffect } from "react";

const PAYOUT_CAP_PCT = 0.05;
const PAYOUT_FLOOR_PCT = 0.001;
const PAYOUT_LOG_SIGMA = 0.6;
// genPayout used cycle probs: P(1)=0.55, P(2)=0.30, P(3)=0.15
const E_CYCLES = 0.55 * 1 + 0.30 * 2 + 0.15 * 3; // 1.60

// Standard normal CDF — Abramowitz & Stegun 7.1.26 (~1.5e-7 accuracy).
function normCdf(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return 0.5 * (1 + sign * y);
}

// Closed-form E[clip(X, lo, hi)] for X ~ LogNormal(mu, sigma²).
// Derivation: split the expectation into lo·P(X<lo) + ∫[lo,hi] x·f(x)dx + hi·P(X>hi).
// The middle integral is exp(mu + σ²/2) · (Φ((ln(hi)−μ−σ²)/σ) − Φ((ln(lo)−μ−σ²)/σ)).
function expectedClippedLognormal(mu, sigma, lo, hi) {
  const zLo = (Math.log(lo) - mu) / sigma;
  const zHi = (Math.log(hi) - mu) / sigma;
  const zLoShift = zLo - sigma;
  const zHiShift = zHi - sigma;
  const pBelow = normCdf(zLo);
  const pAbove = 1 - normCdf(zHi);
  const meanUncapped = Math.exp(mu + 0.5 * sigma * sigma);
  const meanInRange = meanUncapped * (normCdf(zHiShift) - normCdf(zLoShift));
  return lo * pBelow + meanInRange + hi * pAbove;
}

function expectedPayoutPerTrader(accountSize, medianPayoutPct) {
  const mu = Math.log(accountSize * medianPayoutPct);
  const lo = accountSize * PAYOUT_FLOOR_PCT;
  const hi = accountSize * PAYOUT_CAP_PCT;
  const perCycle = expectedClippedLognormal(mu, PAYOUT_LOG_SIGMA, lo, hi);
  return E_CYCLES * perCycle;
}

function runSim(params) {
  const { sizes, platformCost, employeeCost, marketingCost,
    affiliateShare, affiliateComm,
    extraCosts = [] } = params;

  const totalAccounts = sizes.reduce((s, sz) => s + sz.count, 0);

  let grossFees = 0, discounts = 0, affComm = 0, resetRev = 0, payouts = 0;
  let passers = 0, payoutTraders = 0, resetCount = 0;
  const sd = {};

  for (const sz of sizes) {
    const n = sz.count;
    if (n === 0) continue;

    const szPassRate = sz.passRate ?? 10;
    const szFundedPct = sz.fundedPct ?? 10;
    const szAvgPayoutPct = (sz.avgPayoutPct ?? 5) / 100;
    const szResetRate = (sz.resetRate ?? 35) / 100;

    const pc = Math.round(n * szPassRate / 100);
    const fc = n - pc;
    passers += pc;

    const sizeDiscount = (sz.discount ?? 0) / 100;
    const affSales = Math.round(n * affiliateShare);
    const mktSales = n - affSales;
    const gf = n * sz.fee;
    const disc = mktSales * sz.fee * sizeDiscount;
    const ac = affSales * sz.fee * affiliateComm;
    const nf = gf - disc - ac;
    grossFees += gf; discounts += disc; affComm += ac;

    const sizeResetCount = fc * szResetRate;
    const sizeResetPct = (sz.resetPct ?? 80) / 100;
    const effPerReset = sz.fee * sizeResetPct * (1 - affiliateShare * affiliateComm);
    const sizeResetRev = sizeResetCount * effPerReset;
    resetRev += sizeResetRev;
    resetCount += sizeResetCount;

    const sizePT = pc * szFundedPct / 100;
    const sizePaid = sizePT * expectedPayoutPerTrader(sz.size, szAvgPayoutPct);
    payouts += sizePaid;
    payoutTraders += sizePT;

    const sdKey = sz.key || sz.size;
    sd[sdKey] = { gf, disc, ac, nf, resetRev: sizeResetRev, payouts: sizePaid, pt: sizePT, pc, n };
  }

  const totalPlatform = totalAccounts * platformCost;
  const fixed = employeeCost + marketingCost;
  const totalRev = (grossFees - discounts - affComm) + resetRev;
  const extras = computeExtras(extraCosts, {
    totalAccounts, passers, payoutTraders, grossFees, totalRev,
  });
  const totalCost = payouts + totalPlatform + fixed + extras.total;
  const net = totalRev - totalCost;

  return {
    grossFees, discounts, affComm, netFees: grossFees - discounts - affComm,
    resetRev, payouts,
    platform: totalPlatform, fixed,
    extras: extras.total, extrasBreakdown: extras.bd,
    revenue: totalRev, costs: totalCost,
    net, margin: totalRev > 0 ? (net / totalRev) * 100 : 0,
    passers, payoutTraders, resets: resetCount, totalAccounts,
    sizeAvg: sd,
  };
}

const $ = (n) => {
  if (n == null) return "—"; const s = n < 0 ? "-" : ""; const a = Math.abs(n);
  if (a >= 1e6) return `${s}$${(a/1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${s}$${(a/1e3).toFixed(1)}K`;
  return `${s}$${Math.round(a)}`;
};

const fmtNum = (n) => {
  if (n == null || n === "") return "";
  const num = typeof n === "string" ? parseFloat(n) : n;
  if (isNaN(num)) return "";
  if (Number.isInteger(num)) return num.toLocaleString("en-US");
  return num.toLocaleString("en-US", { maximumFractionDigits: 2 });
};

const parseNum = (str) => {
  const cleaned = str.replace(/,/g, "");
  if (cleaned === "" || cleaned === "-") return 0;
  const num = parseFloat(cleaned);
  return isNaN(num) ? 0 : num;
};

// Extra costs — user-definable overhead lines. Each type scales against a
// different simulation quantity, so a single $ amount can represent very
// different real-world costs (flat rent vs. per-funded-trader KYC, etc.).
const COST_TYPES = {
  fixed:       { label: "Fixed $",         short: "$",        desc: "Flat dollar amount" },
  per_account: { label: "$ per account",   short: "$/acct",   desc: "× total accounts sold" },
  per_passer:  { label: "$ per passer",    short: "$/passer", desc: "× traders who pass eval" },
  per_payout:  { label: "$ per payout",    short: "$/payout", desc: "× traders receiving payouts" },
  pct_revenue: { label: "% of revenue",    short: "% rev",    desc: "% of total revenue" },
  pct_fees:    { label: "% of gross fees", short: "% fees",   desc: "% of gross fee revenue" },
};

function computeExtras(extras, ctx) {
  let total = 0;
  const bd = {};
  for (const c of extras) {
    const a = Number(c.amount) || 0;
    let v = 0;
    switch (c.type) {
      case "fixed":       v = a; break;
      case "per_account": v = a * ctx.totalAccounts; break;
      case "per_passer":  v = a * ctx.passers; break;
      case "per_payout":  v = a * ctx.payoutTraders; break;
      case "pct_revenue": v = (a / 100) * ctx.totalRev; break;
      case "pct_fees":    v = (a / 100) * ctx.grossFees; break;
      default: v = 0;
    }
    total += v;
    bd[c.id] = v;
  }
  return { total, bd };
}

const ACCOUNT_SIZES = [
  { size: 2500,   label: "$2.5K" },
  { size: 5000,   label: "$5K" },
  { size: 10000,  label: "$10K" },
  { size: 25000,  label: "$25K" },
  { size: 50000,  label: "$50K" },
  { size: 100000, label: "$100K" },
];

const PROGRAM_FEES = {
  freedom: { 10000: 150, 25000: 400, 50000: 750, 100000: 1165 },
  classic: { 10000: 97, 25000: 225, 50000: 410, 100000: 697 },
  instant: { 2500: 150, 5000: 250, 10000: 400, 25000: 1000, 50000: 2000, 100000: 4000 },
};

const FEE_SIZE_SCHEDULE = [
  { fee: 167,  size: 10000 },
  { fee: 397,  size: 25000 },
  { fee: 747,  size: 50000 },
  { fee: 1197, size: 100000 },
];

function interpolateSize(fee) {
  const s = FEE_SIZE_SCHEDULE;
  if (fee <= s[0].fee) return Math.max(1000, Math.round(s[0].size * (fee / s[0].fee)));
  for (let i = 1; i < s.length; i++) {
    if (fee <= s[i].fee) {
      const t = (fee - s[i - 1].fee) / (s[i].fee - s[i - 1].fee);
      return Math.round(s[i - 1].size + t * (s[i].size - s[i - 1].size));
    }
  }
  const last = s[s.length - 1], prev = s[s.length - 2];
  const t = (fee - prev.fee) / (last.fee - prev.fee);
  return Math.round(prev.size + t * (last.size - prev.size));
}

const Input = ({ label, value, onChange, prefix, suffix, width, small }) => {
  const [editing, setEditing] = useState(false);
  const [raw, setRaw] = useState(String(value));

  const handleFocus = () => {
    setEditing(true);
    setRaw(String(value));
  };

  const handleBlur = () => {
    setEditing(false);
    onChange(parseNum(raw));
  };

  const handleChange = (e) => {
    setRaw(e.target.value);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.target.blur();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <label style={{ fontSize: small ? 9 : 10, color: "#64748b", fontWeight: 600, letterSpacing: "0.03em" }}>{label}</label>
      <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
        {prefix && <span style={{ fontSize: 12, color: "#475569", marginRight: 2 }}>{prefix}</span>}
        <input
          type="text"
          inputMode="decimal"
          value={editing ? raw : fmtNum(value)}
          onChange={handleChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          style={{
            width: width || 80, padding: "5px 6px", background: "rgba(255,255,255,0.06)",
            border: "1px solid rgba(255,255,255,0.1)", borderRadius: 4, color: "#60a5fa",
            fontFamily: "'JetBrains Mono'", fontSize: 12, fontWeight: 600,
          }}
        />
        {suffix && <span style={{ fontSize: 11, color: "#475569", marginLeft: 3 }}>{suffix}</span>}
      </div>
    </div>
  );
};

const presetBtnStyle = {
  padding: "6px 12px", background: "rgba(255,255,255,0.04)", color: "#94a3b8",
  border: "1px solid rgba(255,255,255,0.1)", borderRadius: 4,
  fontWeight: 600, fontSize: 10, cursor: "pointer", letterSpacing: "0.02em",
};

const AccountSizeRow = ({ d, onUpdate, onSizeChange }) => {
  const [editFee, setEditFee] = useState(false);
  const [rawFee, setRawFee] = useState(String(d.fee));
  const [editCount, setEditCount] = useState(false);
  const [rawCount, setRawCount] = useState(String(d.count));

  const cellInput = {
    width: 52, padding: "3px 4px", background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.1)", borderRadius: 3, color: "#60a5fa",
    fontFamily: "'JetBrains Mono'", fontSize: 11, textAlign: "right",
  };

  return (
    <tr>
      <td style={{ padding: "4px" }}>
        <select
          value={d.size}
          onChange={e => onSizeChange(parseInt(e.target.value))}
          style={{
            padding: "3px 2px", background: "rgba(255,255,255,0.06)",
            border: "1px solid rgba(255,255,255,0.1)", borderRadius: 3,
            color: d.color || "#60a5fa", fontFamily: "'JetBrains Mono'",
            fontSize: 11, fontWeight: 700, cursor: "pointer", width: 62,
          }}
        >
          {ACCOUNT_SIZES.map(a => (
            <option key={a.size} value={a.size} style={{ background: "#0f172a" }}>{a.label}</option>
          ))}
        </select>
      </td>
      <td style={{ padding: "4px", textAlign: "right" }}>
        <input
          type="text" inputMode="decimal"
          value={editFee ? rawFee : fmtNum(d.fee)}
          onFocus={() => { setEditFee(true); setRawFee(String(d.fee)); }}
          onBlur={() => { setEditFee(false); onUpdate("fee", parseNum(rawFee)); }}
          onChange={e => setRawFee(e.target.value)}
          onKeyDown={e => e.key === "Enter" && e.target.blur()}
          style={cellInput}
        />
      </td>
      <td style={{ padding: "4px", textAlign: "right" }}>
        <input
          type="text" inputMode="numeric"
          value={editCount ? rawCount : fmtNum(d.count)}
          onFocus={() => { setEditCount(true); setRawCount(String(d.count)); }}
          onBlur={() => { setEditCount(false); onUpdate("count", Math.round(parseNum(rawCount))); }}
          onChange={e => setRawCount(e.target.value)}
          onKeyDown={e => e.key === "Enter" && e.target.blur()}
          style={cellInput}
        />
      </td>
    </tr>
  );
};

const ExtraCostRow = ({ cost, onUpdate, onRemove, impact }) => {
  const [editAmount, setEditAmount] = useState(false);
  const [rawAmount, setRawAmount] = useState(String(cost.amount));
  const isPct = cost.type === "pct_revenue" || cost.type === "pct_fees";

  return (
    <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
      <td style={{ padding: "6px 5px" }}>
        <input
          type="text"
          value={cost.name}
          onChange={e => onUpdate("name", e.target.value)}
          placeholder="Cost name"
          style={{
            width: "100%", minWidth: 140, padding: "5px 7px",
            background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 4, color: "#e2e8f0", fontFamily: "'Inter'",
            fontSize: 11, fontWeight: 500,
          }}
        />
      </td>
      <td style={{ padding: "6px 5px", textAlign: "right" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 3 }}>
          {!isPct && <span style={{ fontSize: 11, color: "#475569" }}>$</span>}
          <input
            type="text"
            inputMode="decimal"
            value={editAmount ? rawAmount : fmtNum(cost.amount)}
            onFocus={() => { setEditAmount(true); setRawAmount(String(cost.amount)); }}
            onBlur={() => { setEditAmount(false); onUpdate("amount", parseNum(rawAmount)); }}
            onChange={e => setRawAmount(e.target.value)}
            onKeyDown={e => e.key === "Enter" && e.target.blur()}
            style={{
              width: 90, padding: "5px 7px", background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.1)", borderRadius: 4, color: "#60a5fa",
              fontFamily: "'JetBrains Mono'", fontSize: 11, fontWeight: 600, textAlign: "right",
            }}
          />
          {isPct && <span style={{ fontSize: 11, color: "#475569" }}>%</span>}
        </div>
      </td>
      <td style={{ padding: "6px 5px" }}>
        <select
          value={cost.type}
          onChange={e => onUpdate("type", e.target.value)}
          title={COST_TYPES[cost.type]?.desc}
          style={{
            padding: "5px 7px", background: "rgba(255,255,255,0.06)",
            border: "1px solid rgba(255,255,255,0.1)", borderRadius: 4, color: "#e2e8f0",
            fontFamily: "'Inter'", fontSize: 11, fontWeight: 500, cursor: "pointer",
          }}
        >
          {Object.entries(COST_TYPES).map(([k, v]) => (
            <option key={k} value={k} style={{ background: "#0f172a" }}>{v.label}</option>
          ))}
        </select>
      </td>
      <td style={{
        padding: "6px 5px", textAlign: "right",
        fontFamily: "'JetBrains Mono'", fontWeight: 700, color: "#ef4444",
      }}>
        {impact != null ? $(impact) : "—"}
      </td>
      <td style={{ padding: "6px 5px", textAlign: "center" }}>
        <button
          onClick={onRemove}
          title="Remove cost"
          style={{
            padding: "3px 9px", background: "rgba(239,68,68,0.1)", color: "#ef4444",
            border: "1px solid rgba(239,68,68,0.2)", borderRadius: 4, cursor: "pointer",
            fontSize: 14, fontWeight: 700, lineHeight: 1,
          }}
        >×</button>
      </td>
    </tr>
  );
};

export default function App() {
  // Programs — each program type has its own set of account sizes,
  // discount %, reset price, and trader performance metrics.
  const [programs, setPrograms] = useState([
    {
      id: "freedom", name: "Freedom Program", color: "#3b82f6",
      discountPct: 15, resetPct: 80,
      passRate: 10, fundedPct: 10, avgPayoutPct: 5, resetRate: 35,
      sizes: [
        { size: 10000,  fee: 150,  count: 250 },
        { size: 25000,  fee: 400,  count: 250 },
        { size: 50000,  fee: 750,  count: 250 },
        { size: 100000, fee: 1165, count: 250 },
      ],
    },
    {
      id: "classic", name: "2 Step Classic", color: "#f59e0b",
      discountPct: 15, resetPct: 80,
      passRate: 10, fundedPct: 10, avgPayoutPct: 5, resetRate: 35,
      sizes: [
        { size: 10000,  fee: 97,   count: 250 },
        { size: 25000,  fee: 225,  count: 250 },
        { size: 50000,  fee: 410,  count: 250 },
        { size: 100000, fee: 697,  count: 250 },
      ],
    },
    {
      id: "instant", name: "Instant Funding", color: "#10b981",
      discountPct: 5, resetPct: 90,
      passRate: 10, fundedPct: 10, avgPayoutPct: 5, resetRate: 35,
      sizes: [
        { size: 2500,   fee: 150,  count: 250 },
        { size: 5000,   fee: 250,  count: 250 },
        { size: 10000,  fee: 400,  count: 250 },
        { size: 25000,  fee: 1000, count: 250 },
        { size: 50000,  fee: 2000, count: 250 },
        { size: 100000, fee: 4000, count: 250 },
      ],
    },
  ]);
  const [activeProgram, setActiveProgram] = useState("freedom");

  // Calc mode: "perSize" (existing per-tier) or "aov" (average order value)
  const [calcMode, setCalcMode] = useState("perSize");
  const [aovAccounts, setAovAccounts] = useState(4000);
  const [aovFee, setAovFee] = useState(500);
  const [aovDiscount, setAovDiscount] = useState(15);
  const [aovResetPct, setAovResetPct] = useState(80);
  const [aovPassRate, setAovPassRate] = useState(10);
  const [aovFundedPct, setAovFundedPct] = useState(10);
  const [aovAvgPayoutPct, setAovAvgPayoutPct] = useState(5);
  const [aovResetRate, setAovResetRate] = useState(35);

  // Multi-month projection
  const [months, setMonths] = useState(1);
  const [growthRate, setGrowthRate] = useState(0);

  // Costs
  const [platformCost, setPlatformCost] = useState(2.75);
  const [employeeCost, setEmployeeCost] = useState(30000);
  const [marketingCost, setMarketingCost] = useState(100000);
  const [affiliateShare, setAffiliateShare] = useState(25);
  const [affiliateComm, setAffiliateComm] = useState(20);

  // Extra user-defined costs / overheads / parameters
  const [extraCosts, setExtraCosts] = useState([]);

  const [results, setResults] = useState(null);
  const [projection, setProjection] = useState(null);

  const updateProgramSize = (progId, sizeIdx, field, val) => {
    setPrograms(prev => prev.map(p =>
      p.id !== progId ? p : {
        ...p,
        sizes: p.sizes.map((s, i) => i === sizeIdx ? { ...s, [field]: val } : s),
      }
    ));
  };
  const handleSizeChange = (progId, sizeIdx, newSize) => {
    const defaultFee = PROGRAM_FEES[progId]?.[newSize] || 0;
    setPrograms(prev => prev.map(p =>
      p.id !== progId ? p : {
        ...p,
        sizes: p.sizes.map((s, i) => i !== sizeIdx ? s : { ...s, size: newSize, fee: defaultFee }),
      }
    ));
  };
  const updateProgram = (progId, field, val) => {
    setPrograms(prev => prev.map(p =>
      p.id === progId ? { ...p, [field]: val } : p
    ));
  };

  const addExtraCost = (preset) => {
    setExtraCosts(prev => [...prev, {
      id: Date.now() + Math.random(),
      name: preset?.name || "New Cost",
      amount: preset?.amount ?? 0,
      type: preset?.type || "fixed",
    }]);
  };
  const updateExtraCost = (id, field, val) => {
    setExtraCosts(prev => prev.map(c => c.id === id ? { ...c, [field]: val } : c));
  };
  const removeExtraCost = (id) => {
    setExtraCosts(prev => prev.filter(c => c.id !== id));
  };

  const aovSize = interpolateSize(aovFee);
  const effectiveSizes = calcMode === "aov"
    ? [{ size: aovSize, fee: aovFee, count: aovAccounts, discount: aovDiscount, resetPct: aovResetPct,
         passRate: aovPassRate, fundedPct: aovFundedPct, avgPayoutPct: aovAvgPayoutPct, resetRate: aovResetRate,
         key: "aov", label: `~$${Math.round(aovSize / 1000)}K`, color: "#10b981" }]
    : programs.flatMap(p => p.sizes.map((s, i) => ({
        ...s,
        discount: p.discountPct,
        resetPct: p.resetPct,
        passRate: p.passRate,
        fundedPct: p.fundedPct,
        avgPayoutPct: p.avgPayoutPct,
        resetRate: p.resetRate,
        key: `${p.id}_${i}`,
        label: ACCOUNT_SIZES.find(a => a.size === s.size)?.label || `$${s.size / 1000}K`,
        color: p.color,
        program: p.name,
      })));
  const totalAccounts = effectiveSizes.reduce((s, d) => s + d.count, 0);

  const run = useCallback(() => {
    const baseParams = {
      sizes: effectiveSizes,
      platformCost, employeeCost, marketingCost,
      affiliateShare: affiliateShare / 100, affiliateComm: affiliateComm / 100,
      extraCosts,
    };

    if (months <= 1) {
      setResults(runSim(baseParams));
      setProjection(null);
    } else {
      const monthly = [];
      let cumNet = 0, cumRev = 0, cumCost = 0;
      for (let m = 0; m < months; m++) {
        const growth = Math.pow(1 + growthRate / 100, m);
        const params = {
          ...baseParams,
          sizes: baseParams.sizes.map(sz => ({ ...sz, count: Math.round(sz.count * growth) })),
        };
        const r = runSim(params);
        cumNet += r.net;
        cumRev += r.revenue;
        cumCost += r.costs;
        monthly.push({ month: m + 1, ...r, cumNet, cumRev, cumCost });
      }
      setResults(monthly[0]);
      setProjection(monthly);
    }
  }, [effectiveSizes, platformCost, employeeCost, marketingCost,
      affiliateShare, affiliateComm, extraCosts, months, growthRate]);

  useEffect(() => { run(); }, []);

  const Row = ({ label, value, color, bold, indent, bg }) => (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: `5px ${indent ? "12px" : "0"} 5px ${indent ? "20px" : "0"}`,
      borderBottom: "1px solid rgba(255,255,255,0.03)", background: bg || "transparent",
    }}>
      <span style={{ fontSize: 11, color: color || "#94a3b8", fontWeight: bold ? 700 : 400 }}>{label}</span>
      <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono'", fontWeight: bold ? 800 : 600, color: color || "#e2e8f0" }}>
        {$(value)}
      </span>
    </div>
  );

  return (
    <div style={{ minHeight: "100vh", background: "#070b14", color: "#e2e8f0", fontFamily: "'Inter', -apple-system, sans-serif", padding: "24px 16px" }}>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet" />

      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        <h1 style={{ fontSize: 20, fontWeight: 800, color: "#f8fafc", margin: "0 0 4px" }}>
          CXM — Full P&L Simulator
        </h1>
        <p style={{ fontSize: 11, color: "#475569", margin: "0 0 16px" }}>
          All inputs editable. 1:30 leverage · 100% split · No daily DD · No consistency rules. Deterministic expected-value model.
        </p>

        {/* ==================== INPUT PANELS ==================== */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>

          {/* Account Distribution */}
          <div style={{ padding: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#3b82f6", margin: 0, letterSpacing: "0.05em", textTransform: "uppercase" }}>Accounts</h3>
              <div style={{ display: "flex", gap: 0, borderRadius: 4, overflow: "hidden", border: "1px solid rgba(255,255,255,0.12)" }}>
                {[["perSize", "Per Size"], ["aov", "AOV"]].map(([k, label]) => (
                  <button
                    key={k}
                    onClick={() => setCalcMode(k)}
                    style={{
                      padding: "4px 10px", fontSize: 9, fontWeight: 700, cursor: "pointer",
                      border: "none", letterSpacing: "0.04em",
                      background: calcMode === k ? "#3b82f6" : "rgba(255,255,255,0.04)",
                      color: calcMode === k ? "#fff" : "#64748b",
                    }}
                  >{label}</button>
                ))}
              </div>
            </div>

            {calcMode === "perSize" ? (
              <>
                {/* Program tabs */}
                <div style={{ display: "flex", gap: 0, marginBottom: 8, borderRadius: 4, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)" }}>
                  {programs.map(p => (
                    <button
                      key={p.id}
                      onClick={() => setActiveProgram(p.id)}
                      style={{
                        flex: 1, padding: "5px 4px", fontSize: 8, fontWeight: 700,
                        cursor: "pointer", border: "none", letterSpacing: "0.03em",
                        background: activeProgram === p.id ? p.color : "rgba(255,255,255,0.04)",
                        color: activeProgram === p.id ? "#fff" : "#64748b",
                      }}
                    >{p.name}</button>
                  ))}
                </div>

                {programs.filter(p => p.id === activeProgram).map(p => (
                  <div key={p.id}>
                    <div style={{ display: "flex", gap: 10, marginBottom: 8 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
                        <span style={{ fontSize: 9, color: "#64748b", fontWeight: 600 }}>Discount</span>
                        <input
                          type="text" inputMode="decimal"
                          value={p.discountPct}
                          onChange={e => updateProgram(p.id, "discountPct", parseNum(e.target.value))}
                          style={{ width: 30, padding: "2px 4px", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 3, color: "#60a5fa", fontFamily: "'JetBrains Mono'", fontSize: 11, textAlign: "right" }}
                        />
                        <span style={{ fontSize: 9, color: "#475569" }}>%</span>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
                        <span style={{ fontSize: 9, color: "#64748b", fontWeight: 600 }}>Reset</span>
                        <input
                          type="text" inputMode="decimal"
                          value={p.resetPct}
                          onChange={e => updateProgram(p.id, "resetPct", parseNum(e.target.value))}
                          style={{ width: 30, padding: "2px 4px", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 3, color: "#60a5fa", fontFamily: "'JetBrains Mono'", fontSize: 11, textAlign: "right" }}
                        />
                        <span style={{ fontSize: 9, color: "#475569" }}>% of fee</span>
                      </div>
                    </div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                      <thead>
                        <tr>
                          {["Size", "Fee", "Qty"].map(h => (
                            <th key={h} style={{ padding: "4px 4px", textAlign: h === "Size" ? "left" : "right", fontSize: 9, color: "#64748b", fontWeight: 700 }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {p.sizes.map((s, i) => (
                          <AccountSizeRow
                            key={`${p.id}_${i}`}
                            d={{ ...s, color: p.color, label: ACCOUNT_SIZES.find(a => a.size === s.size)?.label }}
                            onUpdate={(field, val) => updateProgramSize(p.id, i, field, val)}
                            onSizeChange={(newSize) => handleSizeChange(p.id, i, newSize)}
                          />
                        ))}
                      </tbody>
                    </table>
                    <div style={{ marginTop: 6, fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono'", display: "flex", justifyContent: "space-between" }}>
                      <span>{p.name}:</span>
                      <span style={{ fontWeight: 600 }}>{p.sizes.reduce((s, sz) => s + sz.count, 0).toLocaleString()} accounts</span>
                    </div>
                  </div>
                ))}

                <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid rgba(255,255,255,0.08)", fontSize: 11, color: "#e2e8f0", fontFamily: "'JetBrains Mono'", display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontWeight: 700 }}>Total ({programs.length} programs):</span>
                  <span style={{ fontWeight: 800 }}>{totalAccounts.toLocaleString()} accounts</span>
                </div>
              </>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <Input label="Total Accounts / Month" value={aovAccounts} onChange={setAovAccounts} width={80} />
                <Input label="Avg Order Value (Fee)" value={aovFee} onChange={setAovFee} prefix="$" width={80} />
                <Input label="Marketing Discount" value={aovDiscount} onChange={setAovDiscount} suffix="%" width={50} />
                <Input label="Reset Price" value={aovResetPct} onChange={setAovResetPct} suffix="% of fee" width={50} />
                <div style={{ fontSize: 9, color: "#475569", lineHeight: 1.5, borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 8, marginTop: 2 }}>
                  Interpolated account size: <span style={{ color: "#10b981", fontFamily: "'JetBrains Mono'", fontWeight: 700 }}>${aovSize.toLocaleString()}</span>
                  <br />Uses a single blended tier. Fee ${aovFee} maps to ~${Math.round(aovSize / 1000)}K account via the tier schedule.
                </div>
              </div>
            )}
          </div>

          {/* Trader Performance Metrics */}
          <div style={{ padding: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
            <h3 style={{ fontSize: 11, fontWeight: 700, color: "#f59e0b", margin: "0 0 10px", letterSpacing: "0.05em", textTransform: "uppercase" }}>Trader Performance Metrics</h3>

            {calcMode === "perSize" ? (
              <>
                <div style={{ display: "flex", gap: 0, marginBottom: 8, borderRadius: 4, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)" }}>
                  {programs.map(p => (
                    <button
                      key={p.id}
                      onClick={() => setActiveProgram(p.id)}
                      style={{
                        flex: 1, padding: "5px 4px", fontSize: 8, fontWeight: 700,
                        cursor: "pointer", border: "none", letterSpacing: "0.03em",
                        background: activeProgram === p.id ? p.color : "rgba(255,255,255,0.04)",
                        color: activeProgram === p.id ? "#fff" : "#64748b",
                      }}
                    >{p.name}</button>
                  ))}
                </div>
                {programs.filter(p => p.id === activeProgram).map(p => (
                  <div key={p.id} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    <Input label="Pass Rate" value={p.passRate} onChange={v => updateProgram(p.id, "passRate", v)} suffix="%" width={60} />
                    <Input label="Funded Payout %" value={p.fundedPct} onChange={v => updateProgram(p.id, "fundedPct", v)} suffix="%" width={60} />
                    <Input label="Avg Payout Size" value={p.avgPayoutPct} onChange={v => updateProgram(p.id, "avgPayoutPct", v)} suffix="% of acct" width={60} />
                    <Input label="Reset Rate" value={p.resetRate} onChange={v => updateProgram(p.id, "resetRate", v)} suffix="%" width={60} />
                  </div>
                ))}
              </>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <Input label="Pass Rate" value={aovPassRate} onChange={setAovPassRate} suffix="%" width={60} />
                <Input label="Funded Payout %" value={aovFundedPct} onChange={setAovFundedPct} suffix="%" width={60} />
                <Input label="Avg Payout Size" value={aovAvgPayoutPct} onChange={setAovAvgPayoutPct} suffix="% of acct" width={60} />
                <Input label="Reset Rate" value={aovResetRate} onChange={setAovResetRate} suffix="%" width={60} />
              </div>
            )}

            <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", marginTop: 10, paddingTop: 10 }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: "#f59e0b", marginBottom: 6, letterSpacing: "0.05em", textTransform: "uppercase" }}>Projection</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <Input label="Months" value={months} onChange={v => setMonths(Math.max(1, Math.round(v)))} width={50} />
                <Input label="Growth / Month" value={growthRate} onChange={setGrowthRate} suffix="%" width={50} />
              </div>
            </div>
            <div style={{ fontSize: 9, color: "#475569", marginTop: 6, lineHeight: 1.4 }}>
              Each program has its own performance metrics. Avg Payout capped at 5% per cycle. Set Months &gt; 1 to project.
            </div>
          </div>

          {/* Cost Inputs */}
          <div style={{ padding: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
            <h3 style={{ fontSize: 11, fontWeight: 700, color: "#ef4444", margin: "0 0 10px", letterSpacing: "0.05em", textTransform: "uppercase" }}>Costs & Overheads</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <Input label="Platform / Account" value={platformCost} onChange={setPlatformCost} prefix="$" width={60} />
              <Input label="Employees / Month" value={employeeCost} onChange={setEmployeeCost} prefix="$" width={80} />
              <Input label="Marketing / Month" value={marketingCost} onChange={setMarketingCost} prefix="$" width={80} />
              <Input label="Affiliate Share" value={affiliateShare} onChange={setAffiliateShare} suffix="%" width={50} />
              <Input label="Affiliate Commission" value={affiliateComm} onChange={setAffiliateComm} suffix="%" width={50} />
            </div>
            <div style={{ fontSize: 9, color: "#475569", marginTop: 8, lineHeight: 1.4 }}>
              Discount, reset price, and trader metrics are set per program.
            </div>
          </div>
        </div>

        {/* ==================== EXTRA COSTS ==================== */}
        <div style={{
          padding: 14, marginBottom: 20, background: "rgba(167,139,250,0.03)",
          border: "1px solid rgba(167,139,250,0.15)", borderRadius: 8,
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10, gap: 12, flexWrap: "wrap" }}>
            <div>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#a78bfa", margin: 0, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                Extra Costs & Overheads
              </h3>
              <div style={{ fontSize: 9, color: "#64748b", marginTop: 3, maxWidth: 560 }}>
                Add custom cost lines — rent, SaaS, legal, KYC/onboarding, payment processing, rev-share, tax accrual, etc. Each line scales against its chosen base (flat $, per account, per passer, per payout, % of revenue, or % of fees).
              </div>
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              <button
                onClick={() => addExtraCost({ name: "Rent & Utilities", amount: 5000, type: "fixed" })}
                style={presetBtnStyle}
              >+ Rent</button>
              <button
                onClick={() => addExtraCost({ name: "Payment Processing", amount: 3, type: "pct_fees" })}
                style={presetBtnStyle}
              >+ Payment %</button>
              <button
                onClick={() => addExtraCost({ name: "KYC / Onboarding", amount: 15, type: "per_passer" })}
                style={presetBtnStyle}
              >+ KYC</button>
              <button
                onClick={() => addExtraCost()}
                style={{
                  padding: "6px 14px", background: "rgba(167,139,250,0.2)", color: "#c4b5fd",
                  border: "1px solid rgba(167,139,250,0.4)", borderRadius: 4,
                  fontWeight: 700, fontSize: 11, cursor: "pointer",
                }}
              >+ Add Cost</button>
            </div>
          </div>

          {extraCosts.length === 0 ? (
            <div style={{ fontSize: 11, color: "#475569", padding: "14px 0", textAlign: "center", fontStyle: "italic" }}>
              No extra costs configured. Click a quick-add button or "Add Cost" to start.
            </div>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                  <th style={{ padding: "5px 5px", textAlign: "left", fontSize: 9, color: "#64748b", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}>Name</th>
                  <th style={{ padding: "5px 5px", textAlign: "right", fontSize: 9, color: "#64748b", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}>Amount</th>
                  <th style={{ padding: "5px 5px", textAlign: "left", fontSize: 9, color: "#64748b", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}>Type</th>
                  <th style={{ padding: "5px 5px", textAlign: "right", fontSize: 9, color: "#64748b", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}>Impact (avg)</th>
                  <th style={{ padding: "5px 5px", width: 40 }}></th>
                </tr>
              </thead>
              <tbody>
                {extraCosts.map(c => (
                  <ExtraCostRow
                    key={c.id}
                    cost={c}
                    onUpdate={(f, v) => updateExtraCost(c.id, f, v)}
                    onRemove={() => removeExtraCost(c.id)}
                    impact={results?.extrasBreakdown?.[c.id]}
                  />
                ))}
                <tr style={{ borderTop: "1px solid rgba(255,255,255,0.1)", background: "rgba(167,139,250,0.04)" }}>
                  <td colSpan={3} style={{ padding: "8px 5px", fontSize: 10, fontWeight: 700, color: "#a78bfa", letterSpacing: "0.05em", textTransform: "uppercase" }}>
                    Total Extra Costs ({extraCosts.length})
                  </td>
                  <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 800, color: "#ef4444" }}>
                    {results ? $(results.extras) : "—"}
                  </td>
                  <td></td>
                </tr>
              </tbody>
            </table>
          )}
        </div>

        {/* Run button */}
        <div style={{ marginBottom: 20 }}>
          <button onClick={run}
            style={{
              padding: "10px 32px", background: "#22c55e",
              color: "#000", border: "none", borderRadius: 6,
              fontWeight: 800, fontSize: 13, cursor: "pointer", width: "100%",
            }}>
            {months > 1
              ? `Recalculate — ${totalAccounts.toLocaleString()} Accounts/mo × ${months} Months`
              : `Recalculate — ${totalAccounts.toLocaleString()} Accounts`}
          </button>
        </div>

        {/* ==================== RESULTS ==================== */}
        {results && (() => {
          const projTotals = projection ? {
            net: projection.reduce((s, m) => s + m.net, 0),
            revenue: projection.reduce((s, m) => s + m.revenue, 0),
            costs: projection.reduce((s, m) => s + m.costs, 0),
            totalAccounts: projection.reduce((s, m) => s + m.totalAccounts, 0),
          } : null;
          const heroNet = projTotals ? projTotals.net : results.net;
          const heroRev = projTotals ? projTotals.revenue : results.revenue;
          const heroCosts = projTotals ? projTotals.costs : results.costs;
          const heroAccounts = projTotals ? projTotals.totalAccounts : totalAccounts;
          const heroMargin = heroRev > 0 ? heroNet / heroRev * 100 : 0;
          return (
          <>
            {/* Hero P&L */}
            <div style={{
              padding: 20, marginBottom: 20, borderRadius: 8,
              background: heroNet > 0 ? "rgba(34,197,94,0.05)" : "rgba(239,68,68,0.05)",
              border: `2px solid ${heroNet > 0 ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
            }}>
              <div style={{ fontSize: 10, color: "#64748b", fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                {projection
                  ? `${months}-Month P&L — ${heroAccounts.toLocaleString()} Total Accounts · ${programs.length} Programs`
                  : `Net P&L — ${totalAccounts.toLocaleString()} Accounts · ${calcMode === "perSize" ? `${programs.length} Programs` : "AOV Mode"}`
                }
              </div>
              <div style={{ fontSize: 36, fontWeight: 800, fontFamily: "'JetBrains Mono'", color: heroNet > 0 ? "#22c55e" : "#ef4444", marginTop: 4 }}>
                {$(heroNet)}
              </div>
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
                {heroMargin.toFixed(1)}% margin · Revenue {$(heroRev)} · Costs {$(heroCosts)}
                {projection && ` · ${months} months${growthRate > 0 ? ` @ ${growthRate}%/mo growth` : ""}`}
              </div>
            </div>

            {/* P&L Waterfall */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 24 }}>
              <div style={{ padding: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
                <h3 style={{ fontSize: 11, fontWeight: 700, color: "#22c55e", margin: "0 0 10px", letterSpacing: "0.05em", textTransform: "uppercase" }}>Revenue{projection ? " (Month 1)" : ""}</h3>
                <Row label="Gross Fee Revenue" value={results.grossFees} bold color="#3b82f6" />
                <Row label="Marketing Discounts (per program)" value={-results.discounts} indent color="#ef4444" />
                <Row label={`Affiliate Commissions (${affiliateComm}%)`} value={-results.affComm} indent color="#ef4444" />
                <Row label="= Net Fee Revenue" value={results.netFees} bold bg="rgba(255,255,255,0.03)" />
                <div style={{ height: 6 }} />
                <Row label={`Reset Revenue (${Math.round(results.resets)} resets, per-program pricing)`} value={results.resetRev} color="#6366f1" />
                <div style={{ height: 6 }} />
                <Row label="TOTAL REVENUE" value={results.revenue} bold color="#22c55e" bg="rgba(34,197,94,0.05)" />
              </div>

              <div style={{ padding: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
                <h3 style={{ fontSize: 11, fontWeight: 700, color: "#ef4444", margin: "0 0 10px", letterSpacing: "0.05em", textTransform: "uppercase" }}>Costs{projection ? " (Month 1)" : ""}</h3>
                <Row label={`Trader Payouts (${Math.round(results.payoutTraders)} traders)`} value={results.payouts} color="#ef4444" />
                <div style={{ height: 6 }} />
                <Row label={`Platform ($${platformCost} × ${totalAccounts.toLocaleString()})`} value={results.platform} color="#f59e0b" />
                <Row label="Employees" value={employeeCost} indent color="#f59e0b" />
                <Row label="Marketing" value={marketingCost} indent color="#f59e0b" />
                <Row label="= Fixed + Platform" value={results.fixed + results.platform} bold bg="rgba(255,255,255,0.03)" />
                {extraCosts.length > 0 && (
                  <>
                    <div style={{ height: 6 }} />
                    {extraCosts.map(c => (
                      <Row
                        key={c.id}
                        label={`${c.name || "Extra"} (${COST_TYPES[c.type]?.short || ""})`}
                        value={results.extrasBreakdown?.[c.id] || 0}
                        indent
                        color="#a78bfa"
                      />
                    ))}
                    <Row label="= Extra Costs" value={results.extras} bold color="#a78bfa" bg="rgba(167,139,250,0.05)" />
                  </>
                )}
                <div style={{ height: 6 }} />
                <Row label="TOTAL COSTS" value={results.costs} bold color="#ef4444" bg="rgba(239,68,68,0.05)" />
              </div>
            </div>

            {/* Per-size breakdown */}
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", marginBottom: 8, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                {calcMode === "aov" ? "Blended Tier" : "By Account Size"} (variable costs only — fixed excluded){projection ? " — Month 1" : ""}
              </h3>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr style={{ borderBottom: "2px solid rgba(255,255,255,0.1)" }}>
                      {["Size", "Qty", "Fee", "Gross", "Discounts", "Aff Comm", "Net Fees", "Resets", "Payouts", "Platform", "Variable P&L"].map(h => (
                        <th key={h} style={{ padding: "6px 5px", textAlign: h === "Size" ? "left" : "right", fontSize: 8, fontWeight: 700, color: "#64748b", letterSpacing: "0.05em", textTransform: "uppercase", whiteSpace: "nowrap" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {effectiveSizes.map(d => {
                      const s = results.sizeAvg[d.key || d.size];
                      if (!s) return null;
                      const varPnL = s.nf + s.resetRev - s.payouts - (d.count * platformCost);
                      return (
                        <tr key={d.key || d.size} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                          <td style={{ padding: "7px 5px" }}>
                            {d.program && <div style={{ fontSize: 8, color: "#64748b", lineHeight: 1.2 }}>{d.program}</div>}
                            <div style={{ fontWeight: 700, color: d.color, fontFamily: "'JetBrains Mono'", fontSize: 11 }}>{d.label}</div>
                          </td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#94a3b8" }}>{d.count.toLocaleString()}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#94a3b8" }}>${d.fee}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#94a3b8" }}>{$(s.gf)}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#ef4444" }}>
                            ({$(s.disc)})
                            <span style={{ fontSize: 9, color: "#64748b", marginLeft: 4 }}>{d.discount ?? 0}%</span>
                          </td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#ef4444" }}>({$(s.ac)})</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#3b82f6" }}>{$(s.nf)}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#6366f1" }}>{$(s.resetRev)}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#ef4444" }}>{$(s.payouts)}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#f59e0b" }}>{$(d.count * platformCost)}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 800, color: varPnL > 0 ? "#22c55e" : "#ef4444" }}>{$(varPnL)}</td>
                        </tr>
                      );
                    })}
                    <tr style={{ borderTop: "2px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.03)" }}>
                      <td style={{ padding: "8px 5px", fontWeight: 800 }}>TOTAL</td>
                      <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700 }}>{totalAccounts.toLocaleString()}</td>
                      <td></td>
                      <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700 }}>{$(results.grossFees)}</td>
                      <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700, color: "#ef4444" }}>({$(results.discounts)})</td>
                      <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700, color: "#ef4444" }}>({$(results.affComm)})</td>
                      <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700, color: "#3b82f6" }}>{$(results.netFees)}</td>
                      <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700, color: "#6366f1" }}>{$(results.resetRev)}</td>
                      <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700, color: "#ef4444" }}>{$(results.payouts)}</td>
                      <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700, color: "#f59e0b" }}>{$(results.platform)}</td>
                      <td style={{ padding: "8px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 800, fontSize: 12, color: (results.netFees + results.resetRev - results.payouts - results.platform) > 0 ? "#22c55e" : "#ef4444" }}>
                        {$(results.netFees + results.resetRev - results.payouts - results.platform)}
                      </td>
                    </tr>
                  </tbody>
                </table>
                <div style={{ fontSize: 10, color: "#475569", marginTop: 4 }}>
                  Variable P&L = Net Fees + Resets − Payouts − Platform. Add fixed costs (${((employeeCost + marketingCost) / 1000).toFixed(0)}K){results.extras > 0 ? ` + extra costs (${$(results.extras)})` : ""} for full P&L.
                </div>
              </div>
            </div>

            {/* Key stats (Month 1 / single-month) */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 8, marginBottom: 20 }}>
              {projection && (
                <div style={{ gridColumn: "1 / -1", fontSize: 9, color: "#64748b", fontWeight: 600, marginBottom: -4, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                  Month 1 Unit Economics
                </div>
              )}
              {[
                { l: "Passers", v: `${Math.round(results.passers)}`, c: "#94a3b8", sub: `Per-program pass rates` },
                { l: "Payout Traders", v: `${Math.round(results.payoutTraders)}`, c: "#94a3b8", sub: `Per-program funded %` },
                { l: "Resets Sold", v: `${Math.round(results.resets)}`, c: "#94a3b8", sub: `Per-program reset rates` },
                { l: "Avg Payout", v: results.payoutTraders > 0 ? $(results.payouts / results.payoutTraders) : "—", c: "#94a3b8", sub: "Per funded trader" },
                { l: "Revenue / Account", v: $(results.revenue / totalAccounts), c: "#94a3b8" },
                { l: "Cost / Account", v: $(results.costs / totalAccounts), c: "#94a3b8" },
                { l: "Net / Account", v: $(results.net / totalAccounts), c: results.net > 0 ? "#22c55e" : "#ef4444" },
              ].map(({ l, v, c, sub }) => (
                <div key={l} style={{ padding: "10px 12px", background: "rgba(255,255,255,0.02)", borderLeft: `3px solid ${c}`, borderRadius: "0 6px 6px 0" }}>
                  <div style={{ fontSize: 9, color: "#64748b", fontWeight: 600, letterSpacing: "0.04em" }}>{l}</div>
                  <div style={{ fontSize: 16, fontWeight: 800, color: "#f1f5f9", fontFamily: "'JetBrains Mono'", marginTop: 2 }}>{v}</div>
                  {sub && <div style={{ fontSize: 9, color: "#475569", marginTop: 2 }}>{sub}</div>}
                </div>
              ))}
            </div>

            {/* Monthly projection table */}
            {projection && projection.length > 1 && (
              <div style={{ marginBottom: 24 }}>
                <h3 style={{ fontSize: 11, fontWeight: 700, color: "#3b82f6", marginBottom: 8, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                  Monthly Cash Flow — {months} Months{growthRate > 0 ? ` @ ${growthRate}%/mo Growth` : ""}
                </h3>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead>
                      <tr style={{ borderBottom: "2px solid rgba(255,255,255,0.1)" }}>
                        {["Month", "Accounts", "Revenue", "Costs", "Net P&L", "Cumulative"].map(h => (
                          <th key={h} style={{
                            padding: "6px 8px",
                            textAlign: h === "Month" ? "center" : "right",
                            fontSize: 8, fontWeight: 700, color: "#64748b",
                            letterSpacing: "0.05em", textTransform: "uppercase",
                          }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {projection.map(m => (
                        <tr key={m.month} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                          <td style={{ padding: "7px 8px", textAlign: "center", fontFamily: "'JetBrains Mono'", color: "#94a3b8", fontWeight: 600 }}>{m.month}</td>
                          <td style={{ padding: "7px 8px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#94a3b8" }}>{m.totalAccounts.toLocaleString()}</td>
                          <td style={{ padding: "7px 8px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#22c55e" }}>{$(m.revenue)}</td>
                          <td style={{ padding: "7px 8px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#ef4444" }}>{$(m.costs)}</td>
                          <td style={{ padding: "7px 8px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700, color: m.net > 0 ? "#22c55e" : "#ef4444" }}>{$(m.net)}</td>
                          <td style={{ padding: "7px 8px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 800, color: m.cumNet > 0 ? "#22c55e" : "#ef4444" }}>{$(m.cumNet)}</td>
                        </tr>
                      ))}
                      <tr style={{ borderTop: "2px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.03)" }}>
                        <td style={{ padding: "8px 8px", textAlign: "center", fontWeight: 800 }}>Total</td>
                        <td style={{ padding: "8px 8px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700 }}>{projTotals.totalAccounts.toLocaleString()}</td>
                        <td style={{ padding: "8px 8px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700, color: "#22c55e" }}>{$(projTotals.revenue)}</td>
                        <td style={{ padding: "8px 8px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 700, color: "#ef4444" }}>{$(projTotals.costs)}</td>
                        <td style={{ padding: "8px 8px", textAlign: "right", fontFamily: "'JetBrains Mono'", fontWeight: 800, fontSize: 12, color: projTotals.net > 0 ? "#22c55e" : "#ef4444" }}>{$(projTotals.net)}</td>
                        <td></td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                {(() => {
                  const beMonth = projection.findIndex(m => m.cumNet > 0);
                  return beMonth > 0 ? (
                    <div style={{ fontSize: 10, color: "#22c55e", marginTop: 6, fontWeight: 600 }}>
                      Break-even at month {projection[beMonth].month} (cumulative turns positive at {$(projection[beMonth].cumNet)})
                    </div>
                  ) : null;
                })()}
              </div>
            )}
          </>
          );
        })()}
      </div>
    </div>
  );
}
