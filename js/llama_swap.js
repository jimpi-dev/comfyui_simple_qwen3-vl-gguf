// js/llama_swap.js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "SimpleQwenVLggufV2";
const SWAP_WIDGETS = ["llama_swap_url", "llama_swap_model", "llama_swap_log_lines"];
const PLACEHOLDER = "(none)";

function widgetByName(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function isPlaceholder(name) {
    const val = String(name || "").trim().toLowerCase();
    return !val || val === PLACEHOLDER || val === "none" || val === "(loading...)" || val === "(error)";
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

async function fetchModels(url) {
    const qs = new URLSearchParams({ url });
    const res = await api.fetchApi(`/simpleqwenvl/llama_swap/models?${qs.toString()}`);
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
}

async function fetchLogs(url, model, lines) {
    const qs = new URLSearchParams({
        url,
        model: model || "",
        lines: String(lines || 200),
    });
    const res = await api.fetchApi(`/simpleqwenvl/llama_swap/logs?${qs.toString()}`);
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data.log || "";
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

app.registerExtension({
    name: "SimpleQwenVL.LlamaSwapUI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            const node = this;
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
                const url = (urlW?.value || serverW?.value || "").trim();
                if (!url) return;
                try {
                    const lines = linesW?.value || 200;
                    const model = modelW?.value || "";
                    setLogPreview(node, await fetchLogs(url, model, lines));
                } catch (err) {
                    setLogPreview(node, String(err.message || err));
                }
                node.setDirtyCanvas?.(true, true);
            };

            const refresh = async () => {
                if (!toggle?.value) return;
                const url = (urlW?.value || serverW?.value || "").trim();
                if (!url) {
                    if (modelW) setComboValues(modelW, [PLACEHOLDER]);
                    return;
                }
                if (node._llamaSwapBusy) return;
                node._llamaSwapBusy = true;
                try {
                    const data = await fetchModels(url);
                    if (modelW) {
                        const preferred =
                            (!isPlaceholder(modelW.value) && modelW.value) || data.current;
                        setComboValues(modelW, data.models || [], preferred);
                    }
                    await refreshLogs();
                } catch (err) {
                    if (modelW) {
                        const keep = !isPlaceholder(modelW.value) ? [modelW.value] : [];
                        setComboValues(modelW, keep, modelW.value);
                    }
                    setLogPreview(node, `llama-swap: ${err.message || err}`);
                    console.warn("[SimpleQwenVL] llama-swap refresh failed:", err);
                } finally {
                    node._llamaSwapBusy = false;
                    node.setDirtyCanvas?.(true, true);
                }
            };

            node._refreshLlamaSwap = refresh;
            node._refreshLlamaSwapLogs = refreshLogs;

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
            if (toggle?.value) {
                setTimeout(refresh, 50);
            }
            return ret;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = onConfigure?.apply(this, arguments);
            this._applyLlamaSwapVisibility?.();
            const toggle = widgetByName(this, "use_llama_swap");
            if (toggle?.value) {
                setTimeout(() => this._refreshLlamaSwap?.(), 50);
            }
            return ret;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const log = message?.llama_swap_log;
            if (Array.isArray(log) && log.length) {
                setLogPreview(this, log[0]);
                this.setDirtyCanvas?.(true, true);
            }
        };
    },
});
