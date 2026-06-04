// Format a clock time as 12-hour AM/PM. Accepts a "HH:MM" 24h string, an ISO
// datetime, or anything Date can parse. Returns "" for empty, the raw input if
// it can't be parsed. Examples: "19:17" -> "7:17 PM", "06:18" -> "6:18 AM",
// "00:13" -> "12:13 AM".
export function to12h(t) {
  if (t == null || t === "") return "";
  const s = String(t);
  let h;
  let m;
  const hhmm = s.match(/(\d{1,2}):(\d{2})/); // "HH:MM" or "...THH:MM"
  if (hhmm) {
    h = parseInt(hhmm[1], 10);
    m = hhmm[2];
  } else {
    const d = new Date(s);
    if (Number.isNaN(d.getTime())) return s;
    h = d.getHours();
    m = String(d.getMinutes()).padStart(2, "0");
  }
  const ampm = h >= 12 ? "PM" : "AM";
  h = h % 12 || 12;
  return `${h}:${m} ${ampm}`;
}
