import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query

app = FastAPI(
    title="ContextSLM4AVR API",
    description="Returns a requested number of patches for a given VUL4J id.",
    version="1.0.0",
)

DATA_PATH = Path(__file__).parent / "data" / "patches_V2.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    PATCH_DATA = json.load(f)


@app.get("/health")
def health():
    return {"status": "ok", "ids_loaded": len(PATCH_DATA)}


@app.get("/ids")
def list_ids():
    """Returns the list of available ids."""
    return {"ids": list(PATCH_DATA.keys())}


@app.get("/patches", response_model=List[str])
def get_patches(
    id: str = Query(..., description="The item's identifier, e.g. VUL4J-1"),
    count: int = Query(..., gt=0, description="How many patches to return"),
):
    """
    Returns the patches for the given id, in the requested count.

    - If `count` is less than or equal to the number of available patches,
      the first `count` patches are returned.
    - If `count` is greater than 10 (i.e. greater than the number of
      available patches), the response is padded to the requested length
      by duplicating the last element of the list.
    """
    if id not in PATCH_DATA:
        raise HTTPException(status_code=404, detail=f"Unknown id: {id}")

    patches = PATCH_DATA[id]
    available = len(patches)

    if count <= available:
        return patches[:count]

    # count > available (typically count > 10): pad the response to the
    # requested length by duplicating the last element
    result = list(patches)
    last = patches[-1]
    while len(result) < count:
        result.append(last)

    return result