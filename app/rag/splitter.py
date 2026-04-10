"""文档切分器 - 两层切分策略"""

import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.logging import logger


def split_handbook(text: str) -> list[Document]:
    """两层切分：先按 Markdown 标题切分，再对超长节做字符切分"""

    # 第一层：按 Markdown 标题切分
    headers_to_split = [
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False,
    )
    md_chunks = md_splitter.split_text(text)
    logger.info(f"第一层切分: {len(md_chunks)} 个章节块")

    # 第二层：对超长 chunk（>800 字符）做递归切分
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "；", "，", " "],
        length_function=len,
    )

    # 过短 chunk（纯标题/分隔符）无检索价值，跳过
    MIN_CHUNK_LENGTH = 50

    final_docs = []
    for chunk in md_chunks:
        if len(chunk.page_content.strip()) < MIN_CHUNK_LENGTH:
            logger.debug(f"跳过过短 chunk ({len(chunk.page_content)} chars): "
                         f"{chunk.page_content[:40]!r}")
            continue

        # 构建父标题链
        header_chain = _build_header_chain(chunk.metadata)

        if len(chunk.page_content) > 800:
            sub_chunks = char_splitter.split_text(chunk.page_content)
            for i, sub in enumerate(sub_chunks):
                final_docs.append(Document(
                    page_content=sub,
                    metadata={
                        **chunk.metadata,
                        "header_chain": header_chain,
                        "sub_chunk_index": i,
                    },
                ))
        else:
            final_docs.append(Document(
                page_content=chunk.page_content,
                metadata={
                    **chunk.metadata,
                    "header_chain": header_chain,
                },
            ))

    logger.info(f"最终切分: {len(final_docs)} 个文档块（已过滤 < {MIN_CHUNK_LENGTH} 字符的碎片）")
    return final_docs


def _build_header_chain(metadata: dict) -> str:
    """构建父标题链，如 '第2部分：震时避险 > 家中震时避险 > 卧室（夜间）'"""
    parts = []
    for key in ["h2", "h3"]:
        if key in metadata and metadata[key]:
            parts.append(metadata[key].strip())
    return " > ".join(parts) if parts else "未分类"
