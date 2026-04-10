"""FastAPI 应用入口"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.db import init_db
from app.api.routers import sessions, chat, eval


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    setup_logging()
    logger.info("Initializing database...")
    await init_db()
    logger.info("Application started")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title="地震应急问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 挂载静态文件
static_dir = Path(__file__).parent / "web" / "static"
templates_dir = Path(__file__).parent / "web" / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# 注册路由
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(eval.router)


# ---------- 统一错误处理 ----------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": {"code": "NOT_FOUND", "message": "资源不存在"}},
    )


@app.exception_handler(422)
async def validation_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": {"code": "VALIDATION_ERROR", "message": "请求参数校验失败"}},
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": {"code": "INTERNAL_ERROR", "message": "系统内部错误"}},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": {"code": "INTERNAL_ERROR", "message": "系统内部错误，请稍后重试"}},
    )


# ---------- 页面路由 ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})
