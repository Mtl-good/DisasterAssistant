/**
 * 地震应急问答系统 - 前端逻辑
 */
(function () {
    "use strict";

    // ---------- Markdown 配置 ----------
    if (typeof marked !== "undefined") {
        marked.setOptions({
            breaks: true,
            gfm: true,
        });
    }

    function renderMarkdown(text) {
        if (typeof marked !== "undefined") {
            return marked.parse(text);
        }
        return text.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>");
    }

    function sanitizeHtml(html) {
        const div = document.createElement("div");
        div.innerHTML = html;
        div.querySelectorAll("script, iframe, object, embed, form").forEach(function (el) {
            el.remove();
        });
        div.querySelectorAll("*").forEach(function (el) {
            for (let i = el.attributes.length - 1; i >= 0; i--) {
                const attr = el.attributes[i].name;
                if (attr.startsWith("on")) {
                    el.removeAttribute(attr);
                }
            }
        });
        return div.innerHTML;
    }

    // ---------- DOM ----------
    const btnNew = document.getElementById("btn-new-session");
    const searchInput = document.getElementById("search-input");
    const sessionList = document.getElementById("session-list");
    const chatArea = document.getElementById("chat-area");
    const emptyState = document.getElementById("empty-state");
    const inputArea = document.getElementById("input-area");
    const queryInput = document.getElementById("query-input");
    const btnSend = document.getElementById("btn-send");
    const sessionTitle = document.getElementById("current-session-title");
    const contextMenu = document.getElementById("context-menu");
    const menuRename = document.getElementById("menu-rename");
    const menuDelete = document.getElementById("menu-delete");
    const btnHamburger = document.getElementById("btn-hamburger");
    const sidebar = document.getElementById("sidebar");
    const sidebarOverlay = document.getElementById("sidebar-overlay");

    // ---------- 状态 ----------
    let currentSessionId = null;
    let isStreaming = false;
    let contextMenuSessionId = null;

    // ---------- API 工具 ----------
    async function api(url, options) {
        if (options === undefined) options = {};
        const resp = await fetch(url, {
            headers: { "Content-Type": "application/json" },
            ...options,
        });
        if (!resp.ok) {
            const err = await resp.json().catch(function () { return {}; });
            throw new Error(err?.error?.message || "HTTP " + resp.status);
        }
        return resp.json();
    }

    // ---------- 侧边栏移动端切换 ----------
    function openSidebar() {
        sidebar.classList.add("open");
        sidebarOverlay.classList.add("active");
        document.body.style.overflow = "hidden";
    }

    function closeSidebar() {
        sidebar.classList.remove("open");
        sidebarOverlay.classList.remove("active");
        document.body.style.overflow = "";
    }

    btnHamburger.addEventListener("click", openSidebar);
    sidebarOverlay.addEventListener("click", closeSidebar);

    // ---------- 会话列表 ----------
    async function loadSessions(keyword) {
        const qs = keyword ? "?keyword=" + encodeURIComponent(keyword) : "";
        const data = await api("/api/sessions" + qs);
        renderSessionList(data.items);
    }

    function renderSessionList(items) {
        sessionList.innerHTML = "";
        items.forEach(function (s) {
            const div = document.createElement("div");
            div.className = "session-item" + (s.id === currentSessionId ? " active" : "");
            div.textContent = s.title;
            div.dataset.id = s.id;
            div.setAttribute("role", "listitem");
            div.addEventListener("click", function () {
                switchSession(s.id, s.title);
                closeSidebar();
            });
            div.addEventListener("contextmenu", function (e) {
                e.preventDefault();
                showContextMenu(e, s.id);
            });
            sessionList.appendChild(div);
        });
    }

    // ---------- 新建会话 ----------
    btnNew.addEventListener("click", async function () {
        const data = await api("/api/sessions", { method: "POST", body: JSON.stringify({}) });
        currentSessionId = data.id;
        sessionTitle.textContent = data.title;
        chatArea.innerHTML = "";
        if (emptyState) emptyState.style.display = "none";
        inputArea.style.display = "block";
        queryInput.focus();
        closeSidebar();
        await loadSessions();
    });

    // ---------- 快捷建议 ----------
    document.querySelectorAll(".suggestion-chip").forEach(function (chip) {
        chip.addEventListener("click", async function () {
            const query = chip.dataset.query;
            if (!query) return;

            // 先新建会话
            const data = await api("/api/sessions", { method: "POST", body: JSON.stringify({}) });
            currentSessionId = data.id;
            sessionTitle.textContent = data.title;
            chatArea.innerHTML = "";
            if (emptyState) emptyState.style.display = "none";
            inputArea.style.display = "block";

            // 填入问题并发送
            queryInput.value = query;
            btnSend.disabled = false;
            await loadSessions();
            sendMessage();
        });
    });

    // ---------- 切换会话 ----------
    async function switchSession(id, title) {
        currentSessionId = id;
        sessionTitle.textContent = title || "";
        inputArea.style.display = "block";
        chatArea.innerHTML = "";
        if (emptyState) emptyState.style.display = "none";

        // 标记活跃
        document.querySelectorAll(".session-item").forEach(function (el) {
            el.classList.toggle("active", el.dataset.id === id);
        });

        // 加载历史消息
        const data = await api("/api/sessions/" + id + "/messages?limit=100");
        data.items.forEach(function (msg) {
            appendMessage(msg.role, msg.content);
        });
        scrollToBottom();
        queryInput.focus();
    }

    // ---------- 搜索 ----------
    let searchTimer;
    searchInput.addEventListener("input", function () {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(function () {
            loadSessions(searchInput.value.trim());
        }, 300);
    });

    // ---------- 发送消息 ----------
    btnSend.addEventListener("click", sendMessage);
    queryInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    queryInput.addEventListener("input", function () {
        btnSend.disabled = !queryInput.value.trim();
        // 自动调整高度
        queryInput.style.height = "auto";
        queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + "px";
    });

    async function sendMessage() {
        const text = queryInput.value.trim();
        if (!text || !currentSessionId || isStreaming) return;

        appendMessage("user", text);
        queryInput.value = "";
        queryInput.style.height = "auto";
        btnSend.disabled = true;
        isStreaming = true;

        // 加载动画
        const loadingEl = document.createElement("div");
        loadingEl.className = "loading-dots";
        loadingEl.innerHTML = "<span></span><span></span><span></span>";
        chatArea.appendChild(loadingEl);
        scrollToBottom();

        try {
            const resp = await fetch("/api/chat/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: currentSessionId, query: text }),
            });

            loadingEl.remove();

            // 创建 assistant 消息气泡
            const msgEl = document.createElement("div");
            msgEl.className = "message assistant";
            const mdContent = document.createElement("div");
            mdContent.className = "md-content";
            msgEl.appendChild(mdContent);
            chatArea.appendChild(msgEl);

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let fullText = "";

            while (true) {
                const result = await reader.read();
                if (result.done) break;

                buffer += decoder.decode(result.value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                let eventType = "";
                for (let i = 0; i < lines.length; i++) {
                    var line = lines[i];
                    if (line.startsWith("event: ")) {
                        eventType = line.slice(7).trim();
                    } else if (line.startsWith("data: ")) {
                        var dataStr = line.slice(6);
                        try {
                            var data = JSON.parse(dataStr);
                            if (eventType === "token") {
                                fullText += data.text;
                                mdContent.innerHTML = sanitizeHtml(renderMarkdown(fullText));
                                scrollToBottom();
                            } else if (eventType === "error") {
                                mdContent.textContent = "\u26A0\uFE0F " + (data.message || "\u8BF7\u6C42\u5931\u8D25");
                            }
                        } catch (e) {
                            // 忽略解析错误
                        }
                    }
                }
            }

            // 刷新会话列表（标题可能已更新）
            await loadSessions(searchInput.value.trim());

        } catch (err) {
            loadingEl.remove();
            appendMessage("assistant", "\u26A0\uFE0F \u7F51\u7EDC\u9519\u8BEF: " + err.message);
        }

        isStreaming = false;
        scrollToBottom();
    }

    // ---------- 消息渲染 ----------
    function appendMessage(role, content) {
        if (emptyState) emptyState.style.display = "none";
        const el = document.createElement("div");
        el.className = "message " + role;

        if (role === "assistant") {
            const mdContent = document.createElement("div");
            mdContent.className = "md-content";
            mdContent.innerHTML = sanitizeHtml(renderMarkdown(content));
            el.appendChild(mdContent);
        } else {
            el.textContent = content;
        }

        chatArea.appendChild(el);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // ---------- 右键菜单 ----------
    function showContextMenu(e, sessionId) {
        contextMenuSessionId = sessionId;
        contextMenu.style.display = "block";
        contextMenu.style.left = e.clientX + "px";
        contextMenu.style.top = e.clientY + "px";
    }

    document.addEventListener("click", function () {
        contextMenu.style.display = "none";
    });

    menuRename.addEventListener("click", async function () {
        var newTitle = prompt("\u8F93\u5165\u65B0\u6807\u9898:");
        if (!newTitle || !contextMenuSessionId) return;
        await api("/api/sessions/" + contextMenuSessionId, {
            method: "PATCH",
            body: JSON.stringify({ title: newTitle }),
        });
        await loadSessions(searchInput.value.trim());
        if (contextMenuSessionId === currentSessionId) {
            sessionTitle.textContent = newTitle;
        }
    });

    menuDelete.addEventListener("click", async function () {
        if (!contextMenuSessionId) return;
        if (!confirm("\u786E\u5B9A\u5220\u9664\u8BE5\u4F1A\u8BDD\uFF1F")) return;
        await api("/api/sessions/" + contextMenuSessionId, { method: "DELETE" });
        if (contextMenuSessionId === currentSessionId) {
            currentSessionId = null;
            chatArea.innerHTML = "";
            inputArea.style.display = "none";
            sessionTitle.textContent = "";
            if (emptyState) {
                chatArea.appendChild(emptyState);
                emptyState.style.display = "";
            }
        }
        await loadSessions(searchInput.value.trim());
    });

    // ---------- 初始化 ----------
    loadSessions();
})();
