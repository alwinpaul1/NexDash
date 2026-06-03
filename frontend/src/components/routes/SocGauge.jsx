// Circular SOC gauge for the Routes result view (theme-aware).
// SVG arc (270deg sweep) colored by arrival SOC (green -> amber -> red),
// big arrival % in the center, START / MIN labels under the arc ends.

function socColor(soc) {
  if (soc >= 80) return "#15803d"; // 80-100% deep green
  if (soc >= 60) return "#22c55e"; // 60-80% green
  if (soc >= 40) return "#eab308"; // 40-60% yellow
  if (soc >= 20) return "#f59e0b"; // 20-40% amber
  return "#ef4444"; // <20% red
}

function polar(cx, cy, r, angleDeg) {
  const a = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

function arcPath(cx, cy, r, startDeg, endDeg) {
  const start = polar(cx, cy, r, startDeg);
  const end = polar(cx, cy, r, endDeg);
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${large} 1 ${end.x} ${end.y}`;
}

export default function SocGauge({ arrivalSoc = 0, startSoc = 100, minSoc = 15 }) {
  const size = 200;
  const cx = size / 2;
  const cy = size / 2;
  const r = 84;

  const sweep = 270;
  const startAngle = 135;
  const pct = Math.max(0, Math.min(100, arrivalSoc));
  const endAngle = startAngle + (sweep * pct) / 100;
  const color = socColor(arrivalSoc);

  // Full-sweep arc length (for the entrance stroke-draw). The 270° arc is
  // 3/4 of a full circle's circumference at radius r.
  const fullLen = 2 * Math.PI * r * (sweep / 360);
  const valueLen = fullLen * (pct / 100);

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <defs>
            <linearGradient id="socStroke" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.7" />
              <stop offset="100%" stopColor={color} stopOpacity="1" />
            </linearGradient>
            <filter id="socGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {/* track */}
          <path
            d={arcPath(cx, cy, r, startAngle, startAngle + sweep)}
            fill="none"
            className="stroke-surface"
            strokeWidth={14}
            strokeLinecap="round"
          />
          {/* value arc */}
          {pct > 0 && (
            <path
              d={arcPath(cx, cy, r, startAngle, endAngle)}
              fill="none"
              stroke="url(#socStroke)"
              strokeWidth={14}
              strokeLinecap="round"
              filter="url(#socGlow)"
              strokeDasharray={`${valueLen} ${fullLen}`}
              data-soc-arc=""
              style={{
                animation: "socDraw 0.9s cubic-bezier(0.16,1,0.3,1) forwards",
              }}
            />
          )}
        </svg>
        <style>{`
          @keyframes socDraw {
            from { stroke-dashoffset: ${valueLen}; }
            to { stroke-dashoffset: 0; }
          }
          @media (prefers-reduced-motion: reduce) {
            [data-soc-arc] { animation: none !important; stroke-dashoffset: 0 !important; }
          }
        `}</style>
        {/* center readout */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-[10px] uppercase tracking-[0.12em] font-semibold text-on-surface-variant">
            Arrival
          </span>
          <span
            className="font-headline font-bold leading-none tabular-nums"
            style={{ fontSize: "46px", color }}
          >
            {Math.round(arrivalSoc)}
            <span className="text-2xl align-top">%</span>
          </span>
          <span className="text-[10px] tracking-wide text-on-surface-variant mt-1">State of Charge</span>
        </div>
      </div>

      <div className="flex justify-between w-44 -mt-6">
        <div className="text-center">
          <p className="text-[10px] uppercase tracking-[0.1em] font-semibold text-on-surface-variant">Start</p>
          <p className="text-sm font-headline font-semibold text-on-surface tabular-nums">{Math.round(startSoc)}%</p>
        </div>
        <div className="text-center">
          <p className="text-[10px] uppercase tracking-[0.1em] font-semibold text-on-surface-variant">Min floor</p>
          <p className="text-sm font-headline font-semibold text-on-surface tabular-nums">{Math.round(minSoc)}%</p>
        </div>
      </div>
    </div>
  );
}
