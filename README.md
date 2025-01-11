# YGO Grifter

A FastAPI project that uses BERT to predictr the price of Yu-Gi-Oh! cards. This project uses Docker for containerization and is based on the setup demonstrated in [this YouTube video](https://www.youtube.com/watch?v=vA0C0k72-b4).

## Quick Start

### Testing Locally

1. Run the following command in your terminal to test the application:
   ```bash
   fastapi dev app/server.py
   ```

2. If the command above doesn't work, you may need to install the required FastAPI dependency:
   ```bash
   pip install "fastapi[standard]"
   ```

### Docker Commands

Build and run the application using Docker with the commands below:

1. Build the Docker image:
   ```bash
   docker build -t ygo_grifter .
   ```

2. Run the Docker container:
   ```bash
   docker run --name ygo_container -p 8000:8000 ygo_grifter
   ```

### Notes
- Ensure Docker is installed and running on your system before executing the commands.
- If additional dependencies are needed, update the `requirements.txt` file and rebuild the Docker image.
