from typing import List, Dict, Optional
import requests

OLLAMA_BASE_URL = "http://localhost:11434"

def generate_answer_ollama(
    query: str,
    contexts: List[Dict[str, str]],
    model: str = "mistral-nemo:latest",
    temperature: float = 0.2,
    system_prompt: Optional[str] = None
) -> str:
    """Generate an answer using a locally running Ollama model."""
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the answer cannot be found in the context, say you don't know and attempt a best-effort answer."
        )

    context_texts = []
    for i, c in enumerate(contexts, 1):
        src = c.get('source', 'unknown')
        txt = c.get('text', '')
        context_texts.append(f"--- Context {i} (source: {src}) ---\n{txt}\n")

    user_message = f"{''.join(context_texts)}\nUser question: {query}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 512
        }
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running: run 'ollama serve' in a terminal."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e.response.text}")


def list_ollama_models() -> List[dict]:
    """Fetch all locally installed Ollama models."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        return [
            {
                "name": m.get("name"),
                "size": m.get("size"),
                "modified_at": m.get("modified_at"),
                "model": m.get("model")
            }
            for m in models
        ]
    except Exception as e:
        return []


def generate_answer_openrouter(
    query: str,
    contexts: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 0.2,
    system_prompt: Optional[str] = None
) -> str:
    """Generate an answer using OpenRouter API."""
    if not api_key:
        try:
            from config import OPENROUTER_API_KEY
            api_key = OPENROUTER_API_KEY
        except ImportError:
            import os
            api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OpenRouter API key is not configured. Please check your .env file.")

    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the answer cannot be found in the context, say you don't know and attempt a best-effort answer."
        )

    context_texts = []
    for i, c in enumerate(contexts, 1):
        src = c.get('source', 'unknown')
        txt = c.get('text', '')
        context_texts.append(f"--- Context {i} (source: {src}) ---\n{txt}\n")

    user_message = f"{''.join(context_texts)}\nUser question: {query}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "RAG Studio",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"OpenRouter API error: {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to OpenRouter: {str(e)}")


def generate_answer(
    query: str,
    contexts: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = "mistral-nemo:latest",
    temperature: float = 0.2,
    system_prompt: Optional[str] = None
) -> str:
    """Generate answer using Ollama if the model is installed locally or if it doesn't contain a slash, otherwise use OpenRouter."""
    # If the model name contains a slash, it's likely an OpenRouter model path (e.g. openai/gpt-3.5-turbo)
    if "/" in model:
        return generate_answer_openrouter(
            query=query,
            contexts=contexts,
            api_key=api_key,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt
        )
    
    # Check if the model is locally installed in Ollama
    local_models = list_ollama_models()
    local_model_names = [m["name"] for m in local_models]
    # Check absolute match or strip tag match (e.g., mistral-nemo vs mistral-nemo:latest)
    is_local = model in local_model_names or any(m.split(":")[0] == model.split(":")[0] for m in local_model_names)
    
    if is_local:
        return generate_answer_ollama(
            query=query,
            contexts=contexts,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt
        )
    else:
        # If it's not a local model, check if we can run it via OpenRouter as a fallback
        try:
            return generate_answer_openrouter(
                query=query,
                contexts=contexts,
                api_key=api_key,
                model=model,
                temperature=temperature,
                system_prompt=system_prompt
            )
        except Exception as openrouter_err:
            # If OpenRouter fails or isn't set up, fall back to Ollama and raise the local connection/not-found error
            raise RuntimeError(
                f"Model '{model}' is not installed locally in Ollama and OpenRouter request failed: {openrouter_err}. "
                f"Available local models: {', '.join(local_model_names) if local_model_names else 'None'}. "
                f"Please ensure Ollama is running or configure OpenRouter key in .env."
            )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('query')
    parser.add_argument('--model', default='mistral-nemo:latest')
    args = parser.parse_args()
    print(generate_answer(args.query, [], model=args.model))
