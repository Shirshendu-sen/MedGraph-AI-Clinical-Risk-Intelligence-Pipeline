from pydantic import BaseModel
from typing import Optional

class ProcessedDocument(BaseModel):
    source_file:    str
    file_type:      str
    cleaned_text:   str
    sentences:      list[str]
    sentence_count: int

class ExtractedEntity(BaseModel):
    text:         str
    label:        str
    start_char:   int
    end_char:     int
    sentence_idx: int
    concept_id:   Optional[str]   = None
    concept_name: Optional[str]   = None
    kb_source:    Optional[str]   = None
    link_score:   Optional[float] = None
    timestamp:    Optional[str]   = None
