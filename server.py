# pip install fastapi uvicorn requests
# pip install "fastapi[all]"


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# import requests
# import json
# import re

# app = FastAPI()

# # Allow CORS for all origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allow all headers
# )

# # Serve static files (e.g., index.html) from the /static folder
# app.mount("/static", StaticFiles(directory="static"), name="static")

# class ChatRequest(BaseModel):
#     prompt: str

# OLLAMA_URL = "http://localhost:11434/api/generate"

# # Career counselor system instructions
# SYSTEM_INSTRUCTIONS = """
# You are an expert career counselor with deep knowledge of various professions, industries, and career paths. Your purpose is to help users discover professional paths that align with their interests, skills, values, and goals.

# Guidelines:
# 1. Only respond to questions related to career guidance, professional development, and job search advice.
# 2. For unrelated questions, politely redirect the conversation to career topics.
# 3. Help users identify their strengths, interests, and values to find suitable career paths.
# 4. Provide practical, actionable advice for career transitions and professional growth.
# 5. Be encouraging, supportive, and empathetic in your responses.
# 6. When appropriate, ask clarifying questions to better understand the user's situation.
# 7. Base your recommendations on the user's unique circumstances and preferences.
# 8. Avoid making absolute statements about which path is "best" - focus on helping users make informed decisions.

# Remember: Your goal is to empower users to make informed career decisions that align with their personal and professional aspirations.
# """

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     try:
#         # Combine system instructions with user prompt
#         full_prompt = f"{SYSTEM_INSTRUCTIONS}\n\nUser: {request.prompt}\nCareer Counselor:"
        
#         response = requests.post(
#             OLLAMA_URL, 
#             json={"model": "deepseek-r1", "prompt": full_prompt}, # deepseek-r1:14b
#             stream=True
#         )
        
#         result = ""
#         for line in response.iter_lines():
#             if line:
#                 try:
#                     chunk_data = json.loads(line.decode("utf-8"))
#                     if "response" in chunk_data:
#                         result += chunk_data["response"]
#                 except json.JSONDecodeError:
#                     continue  # Ignore invalid JSON lines
        
#         if not result:
#             raise HTTPException(status_code=500, detail="No response from model")
        
#         # Clean the result by removing tags and extra whitespace
#         clean_result = re.sub(r'<.*?>', '', result)  # Remove all tags (like <think>)
#         clean_result = clean_result.strip()  # Remove extra spaces
#         clean_result = re.sub(r'\n+', ' ', clean_result)  # Replace multiple newlines with space
        
#         return {"response": clean_result}
    
#     except requests.exceptions.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

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

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
