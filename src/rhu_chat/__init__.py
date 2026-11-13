"""RHU Catalog Explorer: a small, grounded, classical-NLP catalog chatbot."""

__all__ = ["RHUChatPipeline"]


def __getattr__(name):  # lazy import keeps `import src.rhu_chat.data` cheap
    if name == "RHUChatPipeline":
        from .pipeline import RHUChatPipeline

        return RHUChatPipeline
    raise AttributeError(name)
