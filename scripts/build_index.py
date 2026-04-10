"""构建 FAISS 向量索引脚本

用法: python scripts/build_index.py [--handbook PATH]
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.core.logging import setup_logging, logger
from app.rag.ingest import ingest_handbook


def main():
    parser = argparse.ArgumentParser(description="构建 FAISS 向量索引")
    parser.add_argument("--handbook", type=str, default=None, help="手册文件路径")
    args = parser.parse_args()

    setup_logging()
    logger.info("开始构建索引...")

    try:
        count = ingest_handbook(args.handbook)
        logger.info(f"索引构建完成！共 {count} 个文档块")
    except Exception as e:
        logger.error(f"索引构建失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
