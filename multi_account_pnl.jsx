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
  const { sizes, passRate, fundedPct, avgPayoutPct, platformCost, employeeCost, marketingCost,
    affiliateShare, affiliateComm, marketingDiscount, resetDiscount, resetRate,
    extraCosts = [] } = params;

  const totalAccounts = sizes.reduce((s, sz) => s + sz.count, 0);

  let grossFees = 0, discounts = 0, affComm = 0, resetRev = 0, payouts = 0;
  let passers = 0, payoutTraders = 0, resetCount = 0;
  const sd = {};

  for (const sz of sizes) {
    const n = sz.count;
    if (n === 0) continue;

    const pc = Math.round(n * passRate / 100);
    const fc = n - pc;
    passers += pc;

    // Fees — deterministic. Marketing discount applies to non-affiliate
    // sales only; affiliate sales pay full price then rebate a commission.
    const affSales = Math.round(n * affiliateShare);
    const mktSales = n - affSales;
    const gf = n * sz.fee;
    const disc = mktSales * sz.fee * marketingDiscount;
    const ac = affSales * sz.fee * affiliateComm;
    const nf = gf - disc - ac;
    grossFees += gf; discounts += disc; affComm += ac;

    // Expected resets: fc Bernoulli(resetRate) trials, each reset has
    // probability affiliateShare of being an affiliate sale (which nets
    // reset fee minus commission).
    const sizeResetCount = fc * resetRate;
    const effPerReset = sz.fee * resetDiscount * (1 - affiliateShare * affiliateComm);
    const sizeResetRev = sizeResetCount * effPerReset;
    resetRev += sizeResetRev;
    resetCount += sizeResetCount;

    // Expected payouts: of the `pc` passers, a fraction `fundedPct` ever
    // collect; each collector earns E[clipped lognormal] · E[cycles].
    const sizePT = pc * fundedPct;
    const sizePaid = sizePT * expectedPayoutPerTrader(sz.size, avgPayoutPct);
    payouts += sizePaid;
    payoutTraders += sizePT;

    sd[sz.size] = { gf, disc, ac, nf, resetRev: sizeResetRev, payouts: sizePaid, pt: sizePT, pc, n };
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
  const [passRate, setPassRate] = useState(10);
  const [fundedPct, setFundedPct] = useState(10);
  const [avgPayoutPct, setAvgPayoutPct] = useState(5); // median per-cycle payout as % of account

  // Account distribution
  const [dist, setDist] = useState([
    { size: 10000, fee: 167, count: 1000, label: "$10K", color: "#06b6d4" },
    { size: 25000, fee: 397, count: 1000, label: "$25K", color: "#3b82f6" },
    { size: 50000, fee: 747, count: 1000, label: "$50K", color: "#8b5cf6" },
    { size: 100000, fee: 1197, count: 1000, label: "$100K", color: "#f59e0b" },
  ]);

  // Costs
  const [platformCost, setPlatformCost] = useState(2.75);
  const [employeeCost, setEmployeeCost] = useState(30000);
  const [marketingCost, setMarketingCost] = useState(100000);
  const [affiliateShare, setAffiliateShare] = useState(25);
  const [affiliateComm, setAffiliateComm] = useState(20);
  const [marketingDiscount, setMarketingDiscount] = useState(15);
  const [resetDiscount, setResetDiscount] = useState(80);
  const [resetRate, setResetRate] = useState(35);

  // Extra user-defined costs / overheads / parameters
  const [extraCosts, setExtraCosts] = useState([]);

  const [results, setResults] = useState(null);

  const updateDist = (idx, field, val) => {
    setDist(prev => prev.map((d, i) => i === idx ? { ...d, [field]: val } : d));
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

  const totalAccounts = dist.reduce((s, d) => s + d.count, 0);

  const run = useCallback(() => {
    setResults(runSim({
      sizes: dist, passRate, fundedPct, avgPayoutPct: avgPayoutPct / 100,
      platformCost, employeeCost, marketingCost,
      affiliateShare: affiliateShare / 100, affiliateComm: affiliateComm / 100,
      marketingDiscount: marketingDiscount / 100, resetDiscount: resetDiscount / 100,
      resetRate: resetRate / 100,
      extraCosts,
    }));
  }, [passRate, fundedPct, avgPayoutPct, dist, platformCost, employeeCost, marketingCost,
      affiliateShare, affiliateComm, marketingDiscount, resetDiscount, resetRate, extraCosts]);

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
          CXM Freedom — Full P&L Simulator
        </h1>
        <p style={{ fontSize: 11, color: "#475569", margin: "0 0 16px" }}>
          All inputs editable. 1:30 leverage · 100% split · No daily DD · No consistency rules. Deterministic expected-value model.
        </p>

        {/* ==================== INPUT PANELS ==================== */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>

          {/* Account Distribution */}
          <div style={{ padding: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
            <h3 style={{ fontSize: 11, fontWeight: 700, color: "#3b82f6", margin: "0 0 10px", letterSpacing: "0.05em", textTransform: "uppercase" }}>Account Distribution</h3>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  {["Size", "Fee", "Qty"].map(h => (
                    <th key={h} style={{ padding: "4px 4px", textAlign: h === "Size" ? "left" : "right", fontSize: 9, color: "#64748b", fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dist.map((d, i) => {
                  const [editFee, setEditFee] = useState(false);
                  const [rawFee, setRawFee] = useState(String(d.fee));
                  const [editCount, setEditCount] = useState(false);
                  const [rawCount, setRawCount] = useState(String(d.count));
                  return (
                  <tr key={d.size}>
                    <td style={{ padding: "4px", fontWeight: 700, color: d.color, fontFamily: "'JetBrains Mono'", fontSize: 11 }}>{d.label}</td>
                    <td style={{ padding: "4px", textAlign: "right" }}>
                      <input type="text" inputMode="decimal"
                        value={editFee ? rawFee : fmtNum(d.fee)}
                        onFocus={() => { setEditFee(true); setRawFee(String(d.fee)); }}
                        onBlur={() => { setEditFee(false); updateDist(i, "fee", parseNum(rawFee)); }}
                        onChange={e => setRawFee(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && e.target.blur()}
                        style={{ width: 60, padding: "3px 4px", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 3, color: "#60a5fa", fontFamily: "'JetBrains Mono'", fontSize: 11, textAlign: "right" }} />
                    </td>
                    <td style={{ padding: "4px", textAlign: "right" }}>
                      <input type="text" inputMode="numeric"
                        value={editCount ? rawCount : fmtNum(d.count)}
                        onFocus={() => { setEditCount(true); setRawCount(String(d.count)); }}
                        onBlur={() => { setEditCount(false); updateDist(i, "count", Math.round(parseNum(rawCount))); }}
                        onChange={e => setRawCount(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && e.target.blur()}
                        style={{ width: 60, padding: "3px 4px", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 3, color: "#60a5fa", fontFamily: "'JetBrains Mono'", fontSize: 11, textAlign: "right" }} />
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{ marginTop: 8, fontSize: 11, color: "#94a3b8", fontFamily: "'JetBrains Mono'", display: "flex", justifyContent: "space-between" }}>
              <span>Total:</span>
              <span style={{ fontWeight: 700 }}>{totalAccounts.toLocaleString()} accounts</span>
            </div>
          </div>

          {/* Trading Params */}
          <div style={{ padding: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
            <h3 style={{ fontSize: 11, fontWeight: 700, color: "#f59e0b", margin: "0 0 10px", letterSpacing: "0.05em", textTransform: "uppercase" }}>Trading Parameters</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <Input label="Pass Rate" value={passRate} onChange={setPassRate} suffix="%" width={60} />
              <Input label="Funded Payout %" value={fundedPct} onChange={setFundedPct} suffix="%" width={60} />
              <Input label="Avg Payout Size" value={avgPayoutPct} onChange={setAvgPayoutPct} suffix="% of acct" width={60} />
              <Input label="Reset Rate" value={resetRate} onChange={setResetRate} suffix="%" width={60} />
              <Input label="Reset Price" value={resetDiscount} onChange={setResetDiscount} suffix="% of fee" width={60} />
            </div>
            <div style={{ fontSize: 9, color: "#475569", marginTop: 6, lineHeight: 1.4 }}>
              Avg Payout Size: median per-cycle payout as % of account. 1.5% = $150 on $10K, $1,500 on $100K. Capped at 5% per cycle.
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
              <Input label="Marketing Discount" value={marketingDiscount} onChange={setMarketingDiscount} suffix="%" width={50} />
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
            {`Recalculate — ${totalAccounts.toLocaleString()} Accounts`}
          </button>
        </div>

        {/* ==================== RESULTS ==================== */}
        {results && (
          <>
            {/* Hero P&L */}
            <div style={{
              padding: 20, marginBottom: 20, borderRadius: 8,
              background: results.net > 0 ? "rgba(34,197,94,0.05)" : "rgba(239,68,68,0.05)",
              border: `2px solid ${results.net > 0 ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
            }}>
              <div style={{ fontSize: 10, color: "#64748b", fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                Net P&L — {totalAccounts.toLocaleString()} Accounts @ {passRate}% Pass / {fundedPct}% Funded Payout
              </div>
              <div style={{ fontSize: 36, fontWeight: 800, fontFamily: "'JetBrains Mono'", color: results.net > 0 ? "#22c55e" : "#ef4444", marginTop: 4 }}>
                {$(results.net)}
              </div>
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
                {results.margin.toFixed(1)}% margin · Revenue {$(results.revenue)} · Costs {$(results.costs)}
              </div>
            </div>

            {/* P&L Waterfall */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 24 }}>
              <div style={{ padding: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
                <h3 style={{ fontSize: 11, fontWeight: 700, color: "#22c55e", margin: "0 0 10px", letterSpacing: "0.05em", textTransform: "uppercase" }}>Revenue</h3>
                <Row label="Gross Fee Revenue" value={results.grossFees} bold color="#3b82f6" />
                <Row label={`Marketing Discounts (${marketingDiscount}%)`} value={-results.discounts} indent color="#ef4444" />
                <Row label={`Affiliate Commissions (${affiliateComm}%)`} value={-results.affComm} indent color="#ef4444" />
                <Row label="= Net Fee Revenue" value={results.netFees} bold bg="rgba(255,255,255,0.03)" />
                <div style={{ height: 6 }} />
                <Row label={`Reset Revenue (${Math.round(results.resets)} resets @ ${resetDiscount}% of fee)`} value={results.resetRev} color="#6366f1" />
                <div style={{ height: 6 }} />
                <Row label="TOTAL REVENUE" value={results.revenue} bold color="#22c55e" bg="rgba(34,197,94,0.05)" />
              </div>

              <div style={{ padding: 14, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8 }}>
                <h3 style={{ fontSize: 11, fontWeight: 700, color: "#ef4444", margin: "0 0 10px", letterSpacing: "0.05em", textTransform: "uppercase" }}>Costs</h3>
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
                By Account Size (variable costs only — fixed costs excluded)
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
                    {dist.map(d => {
                      const s = results.sizeAvg[d.size];
                      if (!s) return null;
                      const varPnL = s.nf + s.resetRev - s.payouts - (d.count * platformCost);
                      return (
                        <tr key={d.size} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                          <td style={{ padding: "7px 5px", fontWeight: 700, color: d.color, fontFamily: "'JetBrains Mono'", fontSize: 11 }}>{d.label}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#94a3b8" }}>{d.count.toLocaleString()}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#94a3b8" }}>${d.fee}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#94a3b8" }}>{$(s.gf)}</td>
                          <td style={{ padding: "7px 5px", textAlign: "right", fontFamily: "'JetBrains Mono'", color: "#ef4444" }}>({$(s.disc)})</td>
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

            {/* Key stats */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 8, marginBottom: 20 }}>
              {[
                { l: "Passers", v: `${Math.round(results.passers)}`, c: "#94a3b8", sub: `${passRate}% of ${totalAccounts.toLocaleString()}` },
                { l: "Payout Traders", v: `${Math.round(results.payoutTraders)}`, c: "#94a3b8", sub: `${fundedPct}% of funded` },
                { l: "Resets Sold", v: `${Math.round(results.resets)}`, c: "#94a3b8", sub: `${resetRate}% of failed` },
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
          </>
        )}
      </div>
    </div>
  );
}
