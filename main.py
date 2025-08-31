# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.endpoints import router
from services.logger_config import setup_logging
from services.config import APP_TITLE
from database.chat_db import Base, async_engine

setup_logging()

app = FastAPI(title=APP_TITLE)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)