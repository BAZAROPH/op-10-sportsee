import langchain_core.documents
from pydantic import BaseModel, Field
from typing import Optional


class ChunkMetadata(BaseModel):
    """
        Structure stricte pour les métadata des docs
    """
    source: str = Field(..., description="Chemin du fichier source")
    filename: str = Field(..., description="Nom du fichier source")
    category: str = Field(default="root", description="Nom du fichier")
    chunk_id_in_doc: int = Field(..., ge=0, description="Position du chunk")
    start_index: int = Field(default=-1, description="Index de départ en caractères")


class DocumentChunkSchema(BaseModel):
    """
        Structure stricte pour un morceau de texte (Chunk) prêt à être indexé
    """
    id: str = Field(..., description="Identifiant unique du chunk")
    text: str = Field(..., min_length=10, description="Le texte (minimum 10 cartactères pour évitrer de trop bruiter nos contexts)")
    metada: ChunkMetadata