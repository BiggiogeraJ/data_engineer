from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from dataeng.sqlagent.config import Config, ModelConfig, ModelProvider

def create_llm(model_config: ModelConfig) -> BaseChatModel:
    """
    Create a language model based on the provided configuration.
    
    Args:
        model_config (ModelConfig): Configuration for the language model.
        
    Returns:
        BaseChatModel: An instance of the configured language model.
    """
    if model_config.provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model=model_config.name,
            temperature=model_config.temperature,
            max_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            verbose=False,
            keep_alive=-1,
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_config.provider}")