"""运行评测脚本

用法: python scripts/run_eval.py --variant G1|G2|G3
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.core.logging import setup_logging, logger
from app.core.db import async_session_factory, init_db
from app.services.eval_service import run_eval, get_eval_report


async def main(variant: str):
    setup_logging()
    await init_db()

    async with async_session_factory() as db:
        logger.info(f"开始评测 variant={variant}...")
        results = await run_eval(db, variant)
        logger.info(f"评测完成，共 {len(results)} 题")

        report = await get_eval_report(db, variant)
        print("\n========== 评测报告 ==========")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 RAG 评测")
    parser.add_argument("--variant", choices=["G1", "G2", "G3"], default="G3")
    args = parser.parse_args()
    asyncio.run(main(args.variant))
