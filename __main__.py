import chainlit as cl

def main():
    # This function is the entry point for the application.
    # You can run it with `python -m src`.
    cl.run(host="0.0.0.0", port=8000, headless=False, watch=True)

if __name__ == "__main__":
    main()
