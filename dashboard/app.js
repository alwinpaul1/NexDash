/* NexDash — Range Check panel wiring.
 *
 * Reads the live Range Check form inputs, POSTs them to /api/predict, and
 * renders the reachability verdict into #range-result. Vanilla JS, no
 * framework. index.html owns all markup/ids; this file owns only the fetch
 * wiring and the result rendering so logic is not duplicated across files.
 */
(function () {
  "use strict";

  // EV Energy Green theme accents (kept here only for inline state styling).
  var COLOR_REACH = "#00d166";
  var COLOR_FAIL = "#ba1a1a";
  var COLOR_ON_SURFACE = "#0b1c30";
  var COLOR_ON_SURFACE_VARIANT = "#3c4a3d";
  var COLOR_SURFACE_LOW = "#eff4ff";

  var PREDICT_URL = "/api/predict";

  /** Read a numeric value from an input by id. Returns null when blank. */
  function readNumber(id) {
    var el = document.getElementById(id);
    if (!el) {
      return null;
    }
    var raw = String(el.value).trim();
    if (raw === "") {
      return null;
    }
    var n = Number(raw);
    return Number.isFinite(n) ? n : NaN;
  }

  /** Round a number to a fixed number of decimals, tolerating non-numbers. */
  function fmt(value, decimals) {
    if (value === null || value === undefined || typeof value !== "number" || !Number.isFinite(value)) {
      return "—";
    }
    var d = decimals === undefined ? 1 : decimals;
    return value.toFixed(d);
  }

  /** Escape text destined for innerHTML to avoid markup injection. */
  function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text === null || text === undefined ? "" : String(text);
    return div.innerHTML;
  }

  /** Replace the content of #range-result with the given HTML string. */
  function setResult(html) {
    var target = document.getElementById("range-result");
    if (target) {
      target.innerHTML = html;
    }
  }

  /** Render an in-progress loading state. */
  function renderLoading() {
    setResult(
      '<div class="rounded-xl border p-4 text-sm" ' +
        'style="border-color:#bbcbb9;background:' + COLOR_SURFACE_LOW + ';color:' + COLOR_ON_SURFACE_VARIANT + '">' +
        '<div class="flex items-center gap-2">' +
        '<span class="material-symbols-outlined animate-spin" style="font-size:20px">progress_activity</span>' +
        '<span>Running range check…</span>' +
        "</div>" +
        "</div>"
    );
  }

  /** Render a human-readable error message. */
  function renderError(message) {
    setResult(
      '<div class="rounded-xl border p-4" ' +
        'style="border-color:' + COLOR_FAIL + ';background:#fff5f5;color:' + COLOR_FAIL + '">' +
        '<div class="flex items-start gap-2">' +
        '<span class="material-symbols-outlined" style="font-size:22px">error</span>' +
        '<div>' +
        '<div class="font-semibold">Range check failed</div>' +
        '<div class="text-sm mt-1" style="color:' + COLOR_ON_SURFACE_VARIANT + '">' +
        escapeHtml(message) +
        "</div>" +
        "</div>" +
        "</div>" +
        "</div>"
    );
  }

  /** Build one labelled metric tile for the verdict card. */
  function metricTile(label, value, unit) {
    return (
      '<div class="rounded-lg p-3" style="background:' + COLOR_SURFACE_LOW + '">' +
      '<div class="text-xs uppercase tracking-wide" style="color:' + COLOR_ON_SURFACE_VARIANT + '">' +
      escapeHtml(label) +
      "</div>" +
      '<div class="text-lg font-semibold mt-1" style="color:' + COLOR_ON_SURFACE + '">' +
      escapeHtml(value) +
      (unit ? ' <span class="text-sm font-normal" style="color:' + COLOR_ON_SURFACE_VARIANT + '">' + escapeHtml(unit) + "</span>" : "") +
      "</div>" +
      "</div>"
    );
  }

  /** Render the reachability verdict returned by /api/predict. */
  function renderVerdict(data) {
    var reaches = Boolean(data.reaches);
    var accent = reaches ? COLOR_REACH : COLOR_FAIL;
    var icon = reaches ? "check_circle" : "cancel";
    var title = reaches ? "REACHES" : "WILL NOT REACH";
    var marginVal = typeof data.margin_kwh === "number" ? data.margin_kwh : null;
    var marginSign = marginVal !== null && marginVal >= 0 ? "+" : "";

    var html =
      '<div class="rounded-xl border-2 p-5" style="border-color:' + accent + ';background:' + COLOR_SURFACE_LOW + '">' +
      // Headline verdict.
      '<div class="flex items-center gap-3">' +
      '<span class="material-symbols-outlined" style="font-size:32px;color:' + accent + '">' + icon + "</span>" +
      '<div>' +
      '<div class="text-xl font-bold" style="color:' + accent + ';font-family:\'Space Grotesk\',sans-serif">' +
      title +
      "</div>" +
      '<div class="text-sm" style="color:' + COLOR_ON_SURFACE_VARIANT + '">Margin ' +
      escapeHtml(marginSign + fmt(marginVal, 1)) +
      " kWh</div>" +
      "</div>" +
      "</div>" +
      // Metric grid.
      '<div class="grid grid-cols-2 gap-3 mt-4">' +
      metricTile("Energy needed", fmt(data.energy_needed_kwh, 1), "kWh") +
      metricTile("Margin", marginSign + fmt(marginVal, 1), "kWh") +
      metricTile("Remaining SOC", fmt(data.remaining_soc_pct, 1), "%") +
      metricTile("Remaining range", fmt(data.remaining_range_km, 0), "km") +
      "</div>";

    if (data.confidence_note) {
      html +=
        '<div class="text-xs mt-4 flex items-start gap-1.5" style="color:' + COLOR_ON_SURFACE_VARIANT + '">' +
        '<span class="material-symbols-outlined" style="font-size:16px">info</span>' +
        "<span>" + escapeHtml(data.confidence_note) + "</span>" +
        "</div>";
    }

    html += "</div>";
    setResult(html);
  }

  /**
   * Collect form inputs into the /api/predict request body. Returns
   * { payload } on success or { error } when a required field is missing or
   * non-numeric.
   */
  function collectPayload() {
    var required = {
      soc_pct: "SOC %",
      distance_km: "Distance (km)",
      payload_t: "Payload (t)",
      speed_kph: "Speed (kph)",
      gradient_pct: "Gradient (%)",
      temperature_c: "Temperature (°C)",
    };

    var payload = {};
    for (var id in required) {
      if (!Object.prototype.hasOwnProperty.call(required, id)) {
        continue;
      }
      var value = readNumber(id);
      if (value === null) {
        return { error: "Please fill in " + required[id] + "." };
      }
      if (Number.isNaN(value)) {
        return { error: required[id] + " must be a number." };
      }
      payload[id] = value;
    }

    // wind_mps is optional; include only when provided and valid.
    var wind = readNumber("wind_mps");
    if (wind !== null) {
      if (Number.isNaN(wind)) {
        return { error: "Wind (m/s) must be a number." };
      }
      payload.wind_mps = wind;
    }

    return { payload: payload };
  }

  /** Parse a fetch Response, surfacing API/HTTP errors as readable messages. */
  function parseResponse(response) {
    return response.text().then(function (text) {
      var body = null;
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

      // Prefer a server-provided detail/message; special-case 503 (no model).
      var detail = body && (body.detail || body.message || body.error);
      if (response.status === 503) {
        throw new Error(
          detail ||
            "Model not available. Run `python run_pipeline.py` to train and save the energy model, then retry."
        );
      }
      throw new Error(detail || "Server returned HTTP " + response.status + ".");
    });
  }

  /** Handle the range-check form submission. */
  function onSubmit(event) {
    event.preventDefault();

    var collected = collectPayload();
    if (collected.error) {
      renderError(collected.error);
      return;
    }

    var button = document.getElementById("range-check-btn");
    if (button) {
      button.disabled = true;
    }
    renderLoading();

    fetch(PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collected.payload),
    })
      .then(parseResponse)
      .then(renderVerdict)
      .catch(function (err) {
        var msg = err && err.message ? err.message : String(err);
        // Network/CORS failures surface as a generic TypeError from fetch.
        if (err instanceof TypeError) {
          msg = "Could not reach the server. Is the API running on this host?";
        }
        renderError(msg);
      })
      .then(function () {
        if (button) {
          button.disabled = false;
        }
      });
  }

  /** Wire the form once the DOM is ready. */
  function init() {
    var form = document.getElementById("range-form");
    if (form) {
      form.addEventListener("submit", onSubmit);
      return;
    }
    // Fallback: wire the Check button directly if no <form> wrapper exists.
    var button = document.getElementById("range-check-btn");
    if (button) {
      button.addEventListener("click", onSubmit);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
