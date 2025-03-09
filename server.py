from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import logging
import re  # For regex-based post-processing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Set up CORS and static files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request model
class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received prompt: {request.prompt}")
        
        # Compose a prompt that instructs DeepSeek‑r1 to only output the final answer,
        # without any internal chain‑of‑thought, explanations, or formatting.
        prompt_text = (
            "Just give the final answer. Do not include any internal chain-of-thought, "
            "explanations, or <think> tags. Simply respond directly and concisely.\n\n"
            f"User: {request.prompt}\nAI:"
        )
        
        logger.info("Sending request to DeepSeek‑r1")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1",
                "prompt": prompt_text,
                "stream": False
            }
        )
        
        logger.info(f"DeepSeek‑r1 response status: {response.status_code}")
        logger.info(f"Response content (first 200 chars): {response.text[:200]}...")
        
        data = response.json()
        if "response" in data:
            result = data["response"]
            # Remove any <think>...</think> sections, ignoring case and extra whitespace
            result = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', result, flags=re.DOTALL | re.IGNORECASE)
            result = result.strip()
            logger.info(f"Extracted final response length: {len(result)}")
            return {"response": result}
        else:
            logger.error(f"Unexpected response format: {data}")
            raise HTTPException(status_code=500, detail="Unexpected response format")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Add this new route to serve index.html at the root URL
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
