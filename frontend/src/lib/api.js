/* NexDash — Range Check API client.
 *
 * POSTs the range-check payload to /api/predict and returns the parsed
 * reachability verdict. Throws Error(message) on any non-OK response,
 * preferring a server-provided detail/message/error when present.
 */

const PREDICT_URL = "/api/predict";

export async function predictRange(payload) {
  let response;
  try {
    response = await fetch(PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    // Network/CORS failures surface as a generic TypeError from fetch.
    throw new Error("Could not reach the server. Is the API running on this host?");
  }

  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (e) {
      body = null;
    }
  }

  if (response.ok) {
    if (body && typeof body === "object") {
      return body;
    }
    throw new Error("Received an unexpected (non-JSON) response from the server.");
  }

  const detail = body && (body.detail || body.message || body.error);
  if (response.status === 503) {
    throw new Error(
      detail ||
        "Model not available. Run `python run_pipeline.py` to train and save the energy model, then retry."
    );
  }
  throw new Error(detail || "Server returned HTTP " + response.status + ".");
}
