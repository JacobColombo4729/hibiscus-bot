import uvicorn

def main():
    # This function is the entry point for the application.
    # It starts the Uvicorn server to run the FastAPI app.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
