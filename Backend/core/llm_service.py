"""LLM Service - Provides AI-powered insights using Ollama or other providers."""

import requests
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str = "ollama"  # ollama, openai, anthropic
    model: str = "llama3.2"  # Default model
    base_url: str = "http://localhost:11434"  # Ollama default
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000


class LLMService:
    """Service for generating AI-powered insights."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        
        # Override with environment variables if set
        self.config.provider = os.getenv("LLM_PROVIDER", self.config.provider)
        self.config.model = os.getenv("LLM_MODEL", self.config.model)
        self.config.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Call Ollama API."""
        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "⚠️ Ollama is not running. Please start Ollama or install it from https://ollama.com"
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    def _call_openai(self, prompt: str, system_prompt: str = "") -> str:
        """Call OpenAI API."""
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.model or "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"OpenAI API Error: {response.json().get('error', {}).get('message', 'Unknown error')}"
                
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using configured LLM provider."""
        if self.config.provider == "ollama":
            return self._call_ollama(prompt, system_prompt)
        elif self.config.provider == "openai":
            return self._call_openai(prompt, system_prompt)
        else:
            return f"Unknown provider: {self.config.provider}"
    
    def check_availability(self) -> Dict[str, Any]:
        """Check if LLM service is available."""
        if self.config.provider == "ollama":
            try:
                response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = [m["name"] for m in response.json().get("models", [])]
                    return {
                        "available": True,
                        "provider": "ollama",
                        "models": models,
                        "current_model": self.config.model
                    }
            except:
                pass
            return {"available": False, "provider": "ollama", "error": "Ollama not running"}
        
        return {"available": True, "provider": self.config.provider}


def create_insights_prompt(summary: Dict, segments: Dict, drivers: List, recommendations: List) -> str:
    """Create a prompt for generating business insights."""
    
    prompt = f"""You are a business analyst helping a retail company understand their customer churn data.

Here is the analysis summary:
- Total customers analyzed: {summary.get('total_customers', 0)}
- Predicted churn rate: {summary.get('churn_rate', 0) * 100:.1f}%
- Customers at high risk: {summary.get('high_risk_count', 0)}
- Revenue at risk: ${summary.get('revenue_at_risk', 0):,.0f}

Risk Segments:
"""
    
    for tier, data in segments.items():
        prompt += f"- {tier.upper()} risk: {data.get('count', 0)} customers, ${data.get('total_clv', 0):,.0f} CLV\n"
    
    prompt += "\nTop churn drivers:\n"
    for i, driver in enumerate(drivers[:5]):
        prompt += f"{i+1}. {driver.get('feature', '').replace('_', ' ')}: {driver.get('insight', '')}\n"
    
    prompt += """
Based on this data, provide:
1. A 2-3 sentence executive summary of the churn situation
2. The single most important action to take right now
3. One specific, actionable recommendation with expected impact

Keep your response concise, business-focused, and actionable. Use bullet points. Avoid technical jargon."""

    return prompt


def create_qa_prompt(question: str, summary: Dict, segments: Dict, drivers: List) -> str:
    """Create a prompt for answering user questions about the data."""
    
    context = f"""You are a helpful business analyst assistant. Answer questions about customer churn analysis.

DATA CONTEXT:
- Total customers: {summary.get('total_customers', 0)}
- Churn rate: {summary.get('churn_rate', 0) * 100:.1f}%
- High risk customers: {summary.get('high_risk_count', 0)}
- Medium risk: {summary.get('medium_risk_count', 0)}
- Low risk: {summary.get('low_risk_count', 0)}
- Revenue at risk: ${summary.get('revenue_at_risk', 0):,.0f}

SEGMENTS:
"""
    
    for tier, data in segments.items():
        context += f"- {tier}: {data.get('count', 0)} customers, avg churn prob {data.get('avg_churn_prob', 0)*100:.0f}%, CLV ${data.get('avg_clv', 0):,.0f}\n"
    
    context += "\nTOP CHURN DRIVERS:\n"
    for driver in drivers[:5]:
        context += f"- {driver.get('feature', '').replace('_', ' ')}: {driver.get('insight', '')}\n"
    
    context += f"""

USER QUESTION: {question}

Provide a helpful, concise answer based on the data above. If the question is outside the scope of this data, say so politely. Keep responses under 150 words."""

    return context


# Singleton instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get or create LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
