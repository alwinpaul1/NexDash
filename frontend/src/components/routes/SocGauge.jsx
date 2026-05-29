// Circular SOC gauge (light theme) for the Routes result view.
// SVG arc (270deg sweep) colored by arrival SOC (green -> amber -> red),
// big arrival % in the center, START / MIN labels under the arc ends.

function socColor(soc) {
  if (soc >= 50) return "#00d166"; // accent
  if (soc >= 25) return "#f59e0b"; // amber
  return "#ba1a1a"; // error
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

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {/* track */}
          <path
            d={arcPath(cx, cy, r, startAngle, startAngle + sweep)}
            fill="none"
            stroke="#e5eeff"
            strokeWidth={14}
            strokeLinecap="round"
          />
          {/* value arc */}
          {pct > 0 && (
            <path
              d={arcPath(cx, cy, r, startAngle, endAngle)}
              fill="none"
              stroke={color}
              strokeWidth={14}
              strokeLinecap="round"
            />
          )}
        </svg>
        {/* center readout */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-[11px] uppercase tracking-wide font-medium text-on-surface-variant">Arrival</span>
          <span
            className="font-headline font-bold leading-none"
            style={{ fontSize: "44px", color }}
          >
            {Math.round(arrivalSoc)}
            <span className="text-2xl align-top">%</span>
          </span>
          <span className="text-[11px] text-on-surface-variant mt-1">State of Charge</span>
        </div>
      </div>

      <div className="flex justify-between w-44 -mt-6">
        <div className="text-center">
          <p className="text-[11px] uppercase tracking-wide font-medium text-on-surface-variant">Start</p>
          <p className="text-sm font-headline font-semibold text-on-surface">{Math.round(startSoc)}%</p>
        </div>
        <div className="text-center">
          <p className="text-[11px] uppercase tracking-wide font-medium text-on-surface-variant">Min</p>
          <p className="text-sm font-headline font-semibold text-on-surface">{Math.round(minSoc)}%</p>
        </div>
      </div>
    </div>
  );
}
