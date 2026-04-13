import { useState, useCallback, useEffect } from "react";

const DD = 0.06;
const LEVERAGE = 30;
const PAYOUT_CAP_PCT = 0.05;
const FUNDED_DAYS = 180;
const BIWEEKLY = 14;

function mulberry32(a) {
  return function() {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  let u = 0, v = 0;
  while (!u) u = rng(); while (!v) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function genPayout(accountSize, rng, avgPayoutPct) {
  const cycleRoll = rng();
  const cycles = cycleRoll < 0.55 ? 1 : cycleRoll < 0.85 ? 2 : 3;
  const cap = accountSize * PAYOUT_CAP_PCT;
  const logMean = Math.log(accountSize * avgPayoutPct);
  const logSigma = 0.6;
  let total = 0;
  for (let c = 0; c < cycles; c++) {
    const raw = Math.exp(logMean + gauss(rng) * logSigma);
    total += Math.max(accountSize * 0.001, Math.min(raw, cap));
  }
  return { total, cycles };
}

function simFundedPhase(accountSize, rng, fundedPayoutPct, avgPayoutPct) {
  if (rng() >= fundedPayoutPct) {
    let eq = accountSize, hwm = accountSize;
    const ls = LEVERAGE / 20;
    for (let d = 1; d <= FUNDED_DAYS; d++) {
      eq *= (1 + -0.0005 + gauss(rng) * 0.016 * ls);
      hwm = Math.max(hwm, eq);
      if (eq <= hwm * (1 - DD)) return { paid: 0, nPay: 0 };
    }
    return { paid: 0, nPay: 0 };
  }
  const p = genPayout(accountSize, rng, avgPayoutPct);
  return { paid: p.total, nPay: p.cycles };
}

function runSim(params, seed, runs = 200) {
  const { sizes, passRate, fundedPct, avgPayoutPct, platformCost, employeeCost, marketingCost,
    affiliateShare, affiliateComm, marketingDiscount, resetDiscount, resetRate } = params;

  const totalAccounts = sizes.reduce((s, sz) => s + sz.count, 0);
  const allRuns = [];

  for (let r = 0; r < runs; r++) {
    const rng = mulberry32(seed + r);
    let grossFees = 0, discounts = 0, affComm = 0, resetRev = 0, payouts = 0;
    let payoutTraders = 0, resetCount = 0, passers = 0;
    const sd = {};

    for (const sz of sizes) {
      const n = sz.count;
      if (n === 0) continue;
      const pc = Math.round(n * passRate / 100);
      const fc = n - pc;
      passers += pc;

      const affSales = Math.round(n * affiliateShare);
      const mktSales = n - affSales;
      const gf = n * sz.fee;
      const disc = mktSales * sz.fee * marketingDiscount;
      const ac = affSales * sz.fee * affiliateComm;
      const nf = gf - disc - ac;

      grossFees += gf; discounts += disc; affComm += ac;

      let sizeResetRev = 0, sizeResetCount = 0;
      for (let i = 0; i < fc; i++) {
        if (rng() < resetRate) {
          const rf = sz.fee * resetDiscount;
          const isAff = rng() < affiliateShare;
          sizeResetRev += isAff ? rf * (1 - affiliateComm) : rf;
          sizeResetCount++;
        }
      }
      resetRev += sizeResetRev; resetCount += sizeResetCount;

      let sizePaid = 0, sizePT = 0;
      for (let i = 0; i < pc; i++) {
        const f = simFundedPhase(sz.size, rng, fundedPct / 100, avgPayoutPct);
        sizePaid += f.paid;
        if (f.nPay > 0) sizePT++;
      }
      payouts += sizePaid; payoutTraders += sizePT;

      sd[sz.size] = { gf, disc, ac, nf, resetRev: sizeResetRev, payouts: sizePaid, pt: sizePT, pc, n };
    }

    const totalPlatform = totalAccounts * platformCost;
    const fixed = employeeCost + marketingCost;
    const totalRev = (grossFees - discounts - affComm) + resetRev;
    const totalCost = payouts + totalPlatform + fixed;
    const net = totalRev - totalCost;

    allRuns.push({ grossFees, discounts, affComm, resetRev, payouts, payoutTraders,
      totalPlatform, fixed, totalRev, totalCost, net, passers, resetCount, sd,
      margin: totalRev > 0 ? net / totalRev * 100 : 0 });
  }

  const avg = (fn) => allRuns.reduce((s, r) => s + fn(r), 0) / runs;
  const sorted = allRuns.map(r => r.net).sort((a, b) => a - b);

  return {
    grossFees: avg(r => r.grossFees), discounts: avg(r => r.discounts),
    affComm: avg(r => r.affComm), netFees: avg(r => r.grossFees) - avg(r => r.discounts) - avg(r => r.affComm),
    resetRev: avg(r => r.resetRev), payouts: avg(r => r.payouts),
    platform: avg(r => r.totalPlatform), fixed: avg(r => r.fixed),
    revenue: avg(r => r.totalRev), costs: avg(r => r.totalCost),
    net: avg(r => r.net), margin: avg(r => r.margin),
    profitable: allRuns.filter(r => r.net > 0).length, total: runs,
    netMin: sorted[0], netMax: sorted[sorted.length - 1],
    net5: sorted[Math.floor(runs * 0.05)], net95: sorted[Math.floor(runs * 0.95)],
    median: sorted[Math.floor(runs / 2)],
    passers: avg(r => r.passers), payoutTraders: avg(r => r.payoutTraders),
    resets: avg(r => r.resetCount), totalAccounts,
    sizeAvg: Object.fromEntries(sizes.map(sz => [sz.size, {
      gf: avg(r => r.sd[sz.size]?.gf || 0), disc: avg(r => r.sd[sz.size]?.disc || 0),
      ac: avg(r => r.sd[sz.size]?.ac || 0), nf: avg(r => r.sd[sz.size]?.nf || 0),
      resetRev: avg(r => r.sd[sz.size]?.resetRev || 0), payouts: avg(r => r.sd[sz.size]?.payouts || 0),
      pt: avg(r => r.sd[sz.size]?.pt || 0), pc: avg(r => r.sd[sz.size]?.pc || 0),
      n: sz.count,
    }])),
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

export default function App() {
  const [seed, setSeed] = useState(42);
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

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const updateDist = (idx, field, val) => {
    setDist(prev => prev.map((d, i) => i === idx ? { ...d, [field]: val } : d));
  };

  const totalAccounts = dist.reduce((s, d) => s + d.count, 0);

  const run = useCallback(() => {
    setLoading(true); setResults(null);
    setTimeout(() => {
      const r = runSim({
        sizes: dist, passRate, fundedPct, avgPayoutPct: avgPayoutPct / 100,
        platformCost, employeeCost, marketingCost,
        affiliateShare: affiliateShare / 100, affiliateComm: affiliateComm / 100,
        marketingDiscount: marketingDiscount / 100, resetDiscount: resetDiscount / 100,
        resetRate: resetRate / 100,
      }, seed);
      setResults(r); setLoading(false);
    }, 150);
  }, [seed, passRate, fundedPct, avgPayoutPct, dist, platformCost, employeeCost, marketingCost,
      affiliateShare, affiliateComm, marketingDiscount, resetDiscount, resetRate]);

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
          All inputs editable. 1:30 leverage · 100% split · No daily DD · No consistency rules. 200 Monte Carlo runs.
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

        {/* Run button */}
        <div style={{ marginBottom: 20 }}>
          <button onClick={run} disabled={loading}
            style={{
              padding: "10px 32px", background: loading ? "#1e293b" : "#22c55e",
              color: loading ? "#64748b" : "#000", border: "none", borderRadius: 6,
              fontWeight: 800, fontSize: 13, cursor: "pointer", width: "100%",
            }}>
            {loading ? "Running 200 simulations…" : `Run Monte Carlo — ${totalAccounts.toLocaleString()} Accounts`}
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
                {results.margin.toFixed(1)}% margin · {results.profitable}/200 profitable · 5th–95th: {$(results.net5)} to {$(results.net95)}
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
                  Variable P&L = Net Fees + Resets − Payouts − Platform. Add fixed costs (${((employeeCost + marketingCost) / 1000).toFixed(0)}K) for full P&L.
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
                { l: "Worst Run (of 200)", v: $(results.netMin), c: results.netMin > 0 ? "#22c55e" : "#ef4444", sub: "Worst single Monte Carlo run" },
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
