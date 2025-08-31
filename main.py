# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.endpoints import router
from services.logger_config import setup_logging
from services.config import APP_TITLE
from database.chat_db import Base, engine

# Create database tables
Base.metadata.create_all(bind=engine)

setup_logging()

app = FastAPI(title=APP_TITLE)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)