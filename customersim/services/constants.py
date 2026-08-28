from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None

