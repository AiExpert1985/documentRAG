# main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.endpoints import router
from services.logger_config import setup_logging # Import the setup function
from services.config import APP_TITLE

# Call the function to configure logging
setup_logging()

app = FastAPI(title=APP_TITLE)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)