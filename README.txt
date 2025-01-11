Used this video for reference: https://www.youtube.com/watch?v=vA0C0k72-b4

Run this in terminal to test:
fastapi dev app/server.py 

If that doesn't work, install this, because I forgot to put it requirements..:
pip install "fastapi[standard]"

Docker Commands that I ripped from Youtube:
docker build -t ygo_grifter .    
docker run --name ygo_container -p 8000:8000 ygo_grifter      
