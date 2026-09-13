"""Database nodes package - database generation with JSON validation"""

from .database_generator import OllamaDatabaseGenerator

NODE_CLASS_MAPPINGS = {
    "OllamaDatabaseGenerator": OllamaDatabaseGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaDatabaseGenerator": "Database Generator",
}
