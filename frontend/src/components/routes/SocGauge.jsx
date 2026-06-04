// Circular SOC gauge for the Routes result view (theme-aware).
// SVG arc (270deg sweep) colored by arrival SOC (green -> amber -> red),
// big arrival % in the center, START / MIN labels under the arc ends.

// Healthy reads as Dayos electric-yellow (the "stop the eye" hue), dropping
// through amber to red so risk stays legible at a glance.
function socColor(soc) {
  if (soc >= 55) return "#fff100"; // healthy — electric yellow (Dayos accent)
  if (soc >= 35) return "#f5b700"; // caution — gold
  if (soc >= 20) return "#f59e0b"; // low — amber
  return "#ef4444"; // critical — red
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
  // Healthy state is the signature lime — bright on dark, but illegible on the
  // light variant's white. So the center numeral uses the theme-aware lime ink
  // (.ck-lime) while the glowing arc keeps the full-bright lime stroke. Caution/
  // low/critical amber+red read fine on both themes, so they use `color` direct.
  const healthy = pct >= 55;

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
            <filter id="socGlow" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="5" result="b" />
              <feMerge>
                <feMergeNode in="b" />
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
          <span className="ck-label text-[10px] font-semibold text-on-surface-variant">
            Arrival
          </span>
          <span
            className={`ck-num font-bold leading-none ${healthy ? "ck-lime ck-glow-lime" : ""}`}
            style={{ fontSize: "46px", ...(healthy ? {} : { color }) }}
          >
            {Math.round(arrivalSoc)}
            <span className="text-2xl align-top">%</span>
          </span>
          <span className="text-[10px] tracking-wide text-on-surface-variant mt-1">State of Charge</span>
        </div>
      </div>

      <div className="flex justify-between w-48 mt-1">
        <div className="text-center">
          <p className="ck-label text-[10px] font-semibold text-on-surface-variant">Start</p>
          <p className="ck-num text-sm font-semibold text-on-surface">{Math.round(startSoc)}%</p>
        </div>
        <div className="text-center">
          <p className="ck-label text-[10px] font-semibold text-on-surface-variant">Min floor</p>
          <p className="ck-num text-sm font-semibold tabular-nums" style={{ color: "#f59e0b" }}>{Math.round(minSoc)}%</p>
        </div>
      </div>
    </div>
  );
}
