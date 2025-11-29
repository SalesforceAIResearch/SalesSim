from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

