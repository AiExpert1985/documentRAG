# main.py
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.endpoints import router
from services.logger_config import setup_logging
from services.config import APP_TITLE
from database.chat_db import Base, async_engine

async def create_db_tables():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def main():
    await create_db_tables()
    setup_logging()
    app = FastAPI(title=APP_TITLE)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.include_router(router)
    return app

if __name__ == "__main__":
    app_instance = asyncio.run(main())
    
    uvicorn.run(app_instance, host="0.0.0.0", port=8000)