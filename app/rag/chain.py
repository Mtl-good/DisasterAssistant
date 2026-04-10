"""RAG 链 - 根据 variant 组装不同链路（G1/G2/G3）"""

import asyncio
from typing import AsyncGenerator

from openai import AsyncOpenAI
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger
from app.rag.prompt import SYSTEM_PROMPT_G3, SYSTEM_PROMPT_BASIC, QUERY_REWRITE_PROMPT
from app.rag.retriever import retrieve
from app.rag.reranker import rerank


class RAGChain:
    """RAG 链封装"""

    def __init__(self, variant: str):
        self.variant = variant
        self.llm = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=60.0,
        )

    async def ainvoke(
        self,
        query: str,
        history: list[tuple[str, str]],
    ) -> dict:
        """非流式执行 RAG 流程，返回完整结果（含检索文档信息）

        Returns:
            {
                "answer": str,
                "sources": [{"section": str, "content": str, "score": float}, ...],
                "rewritten_query": str | None,
            }
        """
        # 1. Query 改写（G2/G3 启用）
        search_query = query
        rewritten_query = None
        if self.variant in ("G2", "G3") and history:
            search_query = await self._rewrite_query(query, history)
            rewritten_query = search_query
            logger.info(f"Query rewritten: {query!r} -> {search_query!r}")

        # 2. 向量检索
        docs = retrieve(search_query, settings.retrieval_top_k)
        logger.info(f"Retrieved {len(docs)} docs for query: {search_query!r}")

        # 3. Reranker（G2/G3 启用）
        if self.variant in ("G2", "G3") and docs:
            docs = await rerank(search_query, docs, settings.rerank_top_n)

        # 记录检索来源
        sources = []
        for doc in docs:
            sources.append({
                "section": doc.metadata.get("header_chain", ""),
                "content": doc.page_content,
                "score": doc.metadata.get("score", doc.metadata.get("rerank_score", 0)),
            })

        # 4. 构造 Prompt
        context = self._build_context(docs)
        if self.variant == "G3":
            system_prompt = SYSTEM_PROMPT_G3.format(context=context)
        else:
            system_prompt = SYSTEM_PROMPT_BASIC.format(context=context)

        # 5. 调用 LLM 非流式生成
        messages = [{"role": "system", "content": system_prompt}]
        for role, content in history:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        logger.info(f"Calling LLM (invoke): model={settings.deepseek_model}, variant={self.variant}")
        resp = await self.llm.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            stream=False,
            temperature=0.1,
            max_tokens=2000,
        )
        answer = resp.choices[0].message.content or ""

        return {
            "answer": answer,
            "sources": sources,
            "rewritten_query": rewritten_query,
        }

    async def astream(
        self,
        query: str,
        history: list[tuple[str, str]],
    ) -> AsyncGenerator[str, None]:
        """执行 RAG 流程并流式返回 token"""

        # 1. Query 改写（G2/G3 启用）
        search_query = query
        if self.variant in ("G2", "G3") and history:
            search_query = await self._rewrite_query(query, history)
            logger.info(f"Query rewritten: {query!r} -> {search_query!r}")

        # 2. 向量检索
        docs = retrieve(search_query, settings.retrieval_top_k)
        logger.info(f"Retrieved {len(docs)} docs for query: {search_query!r}")

        # 3. Reranker（G2/G3 启用）
        if self.variant in ("G2", "G3") and docs:
            docs = await rerank(search_query, docs, settings.rerank_top_n)

        # 4. 构造 Prompt
        context = self._build_context(docs)
        if self.variant == "G3":
            system_prompt = SYSTEM_PROMPT_G3.format(context=context)
        else:
            system_prompt = SYSTEM_PROMPT_BASIC.format(context=context)

        # 5. 调用 LLM 流式生成（携带历史对话）
        messages = [{"role": "system", "content": system_prompt}]
        for role, content in history:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        logger.info(f"Calling LLM: model={settings.deepseek_model}, variant={self.variant}")
        stream = await self.llm.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=2000,
        )
        logger.info("LLM stream started")

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

        logger.info("LLM stream completed")

    async def _rewrite_query(self, query: str, history: list[tuple[str, str]]) -> str:
        """多轮上下文改写"""
        chat_history = ""
        for role, content in history[-10:]:  # 最多取最近 10 条
            label = "用户" if role == "user" else "助手"
            chat_history += f"{label}: {content}\n"

        prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=chat_history.strip(),
            question=query,
        )

        try:
            resp = await self.llm.chat.completions.create(
                model=settings.deepseek_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = resp.choices[0].message.content.strip()
            return rewritten if rewritten else query
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query

    def _build_context(self, docs: list[Document]) -> str:
        """将检索到的文档拼成上下文文本"""
        parts = []
        for i, doc in enumerate(docs, 1):
            header = doc.metadata.get("header_chain", "")
            parts.append(f"[来源{i}] {header}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


def build_chain(variant: str) -> RAGChain:
    """工厂函数：根据 variant 构建 RAG 链"""
    return RAGChain(variant=variant)
