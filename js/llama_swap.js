// js/llama_swap.js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const INFERENCE_NODE = "SimpleQwenVLggufV2";
const UNLOAD_NODE = "SimpleQwenUnload";
const SWAP_WIDGETS = ["llama_swap_url", "llama_swap_model", "llama_swap_log_lines"];
const PLACEHOLDER = "(none)";
const POLL_MS = 4000;

function widgetByName(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function isPlaceholder(name) {
    const val = String(name || "").trim().toLowerCase();
    return !val || val === PLACEHOLDER || val === "none" || val === "(loading...)" || val === "(error)";
}

function resolveUrl(node) {
    const toggle = widgetByName(node, "use_llama_swap");
    const swapUrl = widgetByName(node, "llama_swap_url");
    const serverUrl = widgetByName(node, "server_url");
    if (toggle?.value && String(swapUrl?.value || "").trim()) {
        return String(swapUrl.value).trim();
    }
    return String(serverUrl?.value || swapUrl?.value || "").trim();
}

function setComboValues(widget, values, preferred) {
    const list = values && values.length ? values.slice() : [];
    const keep = preferred || widget.value;
    if (keep && !list.includes(keep) && !isPlaceholder(keep)) {
        list.unshift(keep);
    }
    if (!list.includes(PLACEHOLDER)) {
        list.unshift(PLACEHOLDER);
    }
    widget.options = widget.options || {};
    widget.options.values = list;
    if (preferred && list.includes(preferred) && !isPlaceholder(preferred)) {
        widget.value = preferred;
    } else if (!list.includes(widget.value)) {
        widget.value = PLACEHOLDER;
    }
}

function setHidden(node, names, hidden) {
    for (const name of names) {
        const w = widgetByName(node, name);
        if (!w) continue;
        w.hidden = hidden;
        if (w.element) w.element.hidden = hidden;
        if (w.inputEl) w.inputEl.hidden = hidden;
    }
    const btn = widgetByName(node, "Refresh llama-swap models");
    if (btn) {
        btn.hidden = hidden;
        if (btn.element) btn.element.hidden = hidden;
    }
    const logW = widgetByName(node, "llama_swap_log_preview");
    if (logW) {
        logW.hidden = hidden;
        if (logW.inputEl) logW.inputEl.hidden = hidden;
        if (logW.element) logW.element.hidden = hidden;
    }
    if (node._llamaSwapLogEl) {
        node._llamaSwapLogEl.style.display = hidden ? "none" : "block";
    }
    try {
        node.setSize?.([node.size[0], node.computeSize()[1]]);
    } catch (_e) {
        /* ignore */
    }
    node.setDirtyCanvas?.(true, true);
}

function statusColor(st) {
    if (!st || !st.reachable) return "#ef4444";
    if (st.loading) return "#eab308";
    if (st.loaded) return "#22c55e";
    return "#9ca3af";
}

function statusText(st) {
    if (!st || !st.reachable) return st?.message ? `llama: offline` : "llama: offline";
    if (st.loading) return "llama: loading…";
    if (st.loaded) {
        const name = (st.models && st.models[0]) || (st.running && st.running[0]) || "model";
        return `llama: loaded ${name}`;
    }
    return "llama: idle (no model)";
}

function applyStatus(node, data) {
    const parsed = typeof data === "string" ? (() => {
        try {
            return JSON.parse(data);
        } catch (_e) {
            return { message: data };
        }
    })() : data || {};
    node._llamaStatus = {
        reachable: !!parsed.reachable,
        loaded: !!parsed.loaded,
        loading: !!parsed.loading,
        models: parsed.models || parsed.running || [],
        running: parsed.running || parsed.models || [],
        message: parsed.message || parsed.error || "",
        source: parsed.source || "",
    };
    const color = statusColor(node._llamaStatus);
    const text = statusText(node._llamaStatus);
    if (node._llamaStatusEl) {
        const lamp = node._llamaStatusEl.querySelector(".llama-lamp");
        const label = node._llamaStatusEl.querySelector(".llama-status-text");
        if (lamp) lamp.style.background = color;
        if (label) label.textContent = text;
    }
    node.setDirtyCanvas?.(true, true);
}

function ensureStatusWidget(node) {
    if (node._llamaStatusEl) return widgetByName(node, "llama_status_lamp");
    const el = document.createElement("div");
    el.style.cssText = "display:flex;align-items:center;gap:8px;padding:2px 4px;min-height:22px;";
    el.innerHTML =
        '<span class="llama-lamp" style="width:10px;height:10px;border-radius:50%;background:#6b7280;box-shadow:0 0 6px currentColor;flex:0 0 10px;"></span>' +
        '<span class="llama-status-text" style="font-size:12px;opacity:0.9;">llama: …</span>';
    node._llamaStatusEl = el;
    let w;
    if (typeof node.addDOMWidget === "function") {
        w = node.addDOMWidget("llama_status_lamp", "llama_status_lamp", el, {
            serialize: false,
            hideOnZoom: false,
        });
        if (w) {
            w.computeSize = function (width) {
                return [width, 28];
            };
        }
    }
    applyStatus(node, { reachable: false, loaded: false, message: "offline" });
    return w;
}

async function fetchJson(path) {
    const res = await api.fetchApi(path);
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
}

async function fetchModels(url) {
    const qs = new URLSearchParams({ url });
    return fetchJson(`/simpleqwenvl/llama_swap/models?${qs.toString()}`);
}

async function fetchLogs(url, model, lines) {
    const qs = new URLSearchParams({
        url,
        model: model || "",
        lines: String(lines || 200),
    });
    const data = await fetchJson(`/simpleqwenvl/llama_swap/logs?${qs.toString()}`);
    return data.log || "";
}

async function fetchStatus(url) {
    const qs = new URLSearchParams({ url });
    return fetchJson(`/simpleqwenvl/llama_swap/status?${qs.toString()}`);
}

async function postUnload(url, model) {
    const res = await api.fetchApi("/simpleqwenvl/llama_swap/unload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, model: model || "" }),
    });
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
}

function setLogPreview(node, text) {
    const value = text == null ? "" : String(text);
    if (node._llamaSwapLogEl) {
        node._llamaSwapLogEl.value = value;
        node._llamaSwapLogEl.scrollTop = node._llamaSwapLogEl.scrollHeight;
    }
    const preview = widgetByName(node, "llama_swap_log_preview");
    if (preview) preview.value = value;
}

function ensurePreviewWidget(node) {
    let w = widgetByName(node, "llama_swap_log_preview");
    if (w && node._llamaSwapLogEl) return w;
    const el = document.createElement("textarea");
    el.readOnly = true;
    el.placeholder = "llama-swap log (last N lines)";
    el.spellcheck = false;
    el.style.cssText = [
        "width:100%",
        "height:160px",
        "min-height:120px",
        "resize:vertical",
        "box-sizing:border-box",
        "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
        "font-size:11px",
        "line-height:1.35",
        "background:#1e1e1e",
        "color:#d4d4d4",
        "border:1px solid #444",
        "border-radius:4px",
        "padding:6px",
        "white-space:pre",
        "overflow:auto",
    ].join(";");
    node._llamaSwapLogEl = el;
    if (typeof node.addDOMWidget === "function") {
        w = node.addDOMWidget("llama_swap_log_preview", "llama_swap_log_preview", el, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return el.value;
            },
            setValue(v) {
                el.value = v ?? "";
            },
        });
        if (w) {
            w.computeSize = function (width) {
                if (w.hidden) return [width, 0];
                return [width, 170];
            };
        }
        return w;
    }
    w = node.addWidget("text", "llama_swap_log_preview", "", () => {}, {
        multiline: true,
        serialize: false,
    });
    if (w) {
        if (w.inputEl) {
            w.inputEl.readOnly = true;
            w.inputEl.placeholder = "llama-swap log (last N lines)";
        }
        w.computeSize = function (width) {
            return [width, 140];
        };
    }
    return w;
}

function drawTitleLamp(node, ctx) {
    const st = node._llamaStatus;
    const color = statusColor(st);
    const titleH = (window.LiteGraph && LiteGraph.NODE_TITLE_HEIGHT) || 30;
    const r = 5.5;
    const x = node.size[0] - 14;
    const y = -titleH * 0.5;
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = st?.loaded ? 8 : 0;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
}

function attachStatusPolling(node) {
    const poll = async () => {
        const url = resolveUrl(node);
        if (!url) {
            applyStatus(node, { reachable: false, loaded: false, message: "offline" });
            return;
        }
        try {
            applyStatus(node, await fetchStatus(url));
        } catch (err) {
            applyStatus(node, { reachable: false, loaded: false, message: String(err.message || err) });
        }
    };
    node._pollLlamaStatus = poll;
    if (node._llamaPollTimer) clearInterval(node._llamaPollTimer);
    node._llamaPollTimer = setInterval(poll, POLL_MS);
    setTimeout(poll, 80);
}

function parseExecutedStatus(message) {
    const raw = message?.llama_status;
    if (!raw) return null;
    if (typeof raw === "string") return raw;
    if (Array.isArray(raw) && raw.length) return raw[0];
    if (typeof raw === "object") return raw;
    return null;
}

function hookNodeType(nodeType, kind) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const ret = onNodeCreated?.apply(this, arguments);
        const node = this;
        ensureStatusWidget(node);

        if (kind === "inference") {
            const toggle = widgetByName(node, "use_llama_swap");
            const urlW = widgetByName(node, "llama_swap_url");
            const modelW = widgetByName(node, "llama_swap_model");
            const linesW = widgetByName(node, "llama_swap_log_lines");
            const serverW = widgetByName(node, "server_url");
            node._llamaSwapBusy = false;

            if (modelW && modelW.value && !isPlaceholder(modelW.value)) {
                setComboValues(modelW, [modelW.value], modelW.value);
            }

            const refreshLogs = async () => {
                if (!toggle?.value) return;
                const url = resolveUrl(node);
                if (!url) return;
                try {
                    setLogPreview(node, await fetchLogs(url, modelW?.value || "", linesW?.value || 200));
                } catch (err) {
                    setLogPreview(node, String(err.message || err));
                }
                node.setDirtyCanvas?.(true, true);
            };

            const refresh = async () => {
                const url = resolveUrl(node);
                if (!url) {
                    if (modelW) setComboValues(modelW, [PLACEHOLDER]);
                    return;
                }
                if (node._llamaSwapBusy) return;
                node._llamaSwapBusy = true;
                try {
                    if (toggle?.value) {
                        const data = await fetchModels(url);
                        if (modelW) {
                            const preferred =
                                (!isPlaceholder(modelW.value) && modelW.value) || data.current;
                            setComboValues(modelW, data.models || [], preferred);
                        }
                        applyStatus(node, data);
                        await refreshLogs();
                    }
                    await node._pollLlamaStatus?.();
                } catch (err) {
                    if (modelW) {
                        const keep = !isPlaceholder(modelW.value) ? [modelW.value] : [];
                        setComboValues(modelW, keep, modelW.value);
                    }
                    setLogPreview(node, `llama-swap: ${err.message || err}`);
                    applyStatus(node, { reachable: false, loaded: false, message: String(err.message || err) });
                    console.warn("[SimpleQwenVL] llama-swap refresh failed:", err);
                } finally {
                    node._llamaSwapBusy = false;
                    node.setDirtyCanvas?.(true, true);
                }
            };

            node._refreshLlamaSwap = refresh;
            node._refreshLlamaSwapLogs = refreshLogs;

            if (!widgetByName(node, "Unload llama model (free VRAM)")) {
                node.addWidget("button", "Unload llama model (free VRAM)", null, async () => {
                    const url = resolveUrl(node);
                    if (!url) {
                        applyStatus(node, { reachable: false, loaded: false, message: "no URL" });
                        return;
                    }
                    try {
                        const model = toggle?.value && !isPlaceholder(modelW?.value) ? modelW.value : "";
                        const data = await postUnload(url, model);
                        applyStatus(node, data);
                        if (toggle?.value) await refreshLogs();
                    } catch (err) {
                        applyStatus(node, { reachable: true, loaded: true, message: String(err.message || err) });
                        setLogPreview(node, `unload failed: ${err.message || err}`);
                        console.warn("[SimpleQwenVL] unload failed:", err);
                    }
                });
            }
            if (!widgetByName(node, "Refresh llama-swap models")) {
                node.addWidget("button", "Refresh llama-swap models", null, () => refresh());
            }
            ensurePreviewWidget(node);

            const applyVisibility = () => {
                const on = !!toggle?.value;
                setHidden(node, SWAP_WIDGETS, !on);
            };
            node._applyLlamaSwapVisibility = applyVisibility;

            if (toggle) {
                const orig = toggle.callback;
                toggle.callback = function () {
                    orig?.apply(this, arguments);
                    applyVisibility();
                    if (toggle.value) {
                        if (urlW && !String(urlW.value || "").trim() && serverW?.value) {
                            urlW.value = serverW.value;
                        }
                        refresh();
                    }
                    node._pollLlamaStatus?.();
                };
            }
            if (urlW) {
                const orig = urlW.callback;
                urlW.callback = function () {
                    orig?.apply(this, arguments);
                    if (toggle?.value) {
                        clearTimeout(node._llamaSwapTimer);
                        node._llamaSwapTimer = setTimeout(refresh, 400);
                    }
                    node._pollLlamaStatus?.();
                };
            }
            if (serverW) {
                const orig = serverW.callback;
                serverW.callback = function () {
                    orig?.apply(this, arguments);
                    node._pollLlamaStatus?.();
                };
            }
            if (linesW) {
                const orig = linesW.callback;
                linesW.callback = function () {
                    orig?.apply(this, arguments);
                    if (toggle?.value) {
                        clearTimeout(node._llamaSwapLogTimer);
                        node._llamaSwapLogTimer = setTimeout(refreshLogs, 300);
                    }
                };
            }

            applyVisibility();
            if (toggle?.value) setTimeout(refresh, 50);
        } else {
            const serverW = widgetByName(node, "server_url");
            if (serverW) {
                const orig = serverW.callback;
                serverW.callback = function () {
                    orig?.apply(this, arguments);
                    node._pollLlamaStatus?.();
                };
            }
        }

        attachStatusPolling(node);
        return ret;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const ret = onConfigure?.apply(this, arguments);
        this._applyLlamaSwapVisibility?.();
        const toggle = widgetByName(this, "use_llama_swap");
        if (toggle?.value) setTimeout(() => this._refreshLlamaSwap?.(), 50);
        setTimeout(() => this._pollLlamaStatus?.(), 80);
        return ret;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        const preview = widgetByName(this, "llama_swap_log_preview");
        const log = message?.llama_swap_log;
        if (preview && Array.isArray(log) && log.length) {
            setLogPreview(this, log[0]);
        }
        const st = parseExecutedStatus(message);
        if (st) applyStatus(this, st);
        else this._pollLlamaStatus?.();
        this.setDirtyCanvas?.(true, true);
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
        onDrawForeground?.apply(this, arguments);
        try {
            drawTitleLamp(this, ctx);
        } catch (_e) {
            /* ignore */
        }
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
        if (this._llamaPollTimer) {
            clearInterval(this._llamaPollTimer);
            this._llamaPollTimer = null;
        }
        onRemoved?.apply(this, arguments);
    };
}

app.registerExtension({
    name: "SimpleQwenVL.LlamaSwapUI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === INFERENCE_NODE) hookNodeType(nodeType, "inference");
        if (nodeData.name === UNLOAD_NODE) hookNodeType(nodeType, "unload");
    },
});
