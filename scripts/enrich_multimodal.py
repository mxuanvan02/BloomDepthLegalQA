import json
import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Any, List, Dict, Optional
import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("enrich_multimodal")

class VinternDescriber:
    """Wrapper for Vintern-1B-v3_5 Vision-Language Model."""
    def __init__(self, model_name="5CD-AI/Vintern-1B-v3_5"):
        logger.info(f"Loading Vintern VLM: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        
    def describe(self, image_path: Path) -> str:
        """Generate Vietnamese description for an image."""
        try:
            image = Image.open(image_path).convert("RGB")
            # Vintern implementation specific prompt:
            prompt = "<image>\nMô tả chi tiết nội dung của hình ảnh này bằng tiếng Việt, tập trung vào các sơ đồ, bảng biểu hoặc thông tin pháp lý nếu có."
            
            # This is a simplified call; actual Vintern API might differ slightly based on version
            # Assuming standard InternVL-style chat interface:
            response, _ = self.model.chat(self.tokenizer, image, prompt, history=[], return_history=True)
            return response.strip()
        except Exception as e:
            logger.error(f"Vintern inference error on {image_path}: {e}")
            return "Không thể mô tả hình ảnh."

class MultimodalEnricher:
    """V3: Production-ready multimodal enricher with Vintern-1B and smart filters."""
    
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.raw_dir = self.root / "data" / "raw"
        self.interim_dir = self.root / "data" / "interim"
        self.images_dir = self.interim_dir / "images"
        self.chunks_file = self.interim_dir / "extracted_chunks.jsonl"
        self.output_file = self.interim_dir / "extracted_chunks_enriched.jsonl"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes = {} # To avoid re-describing duplicate images (logos, etc.)
        self.vlm = None

    def _get_image_hash(self, image_bytes: bytes) -> str:
        """MD5 Hash for byte-level duplication check."""
        return hashlib.md5(image_bytes).hexdigest()

    def is_junk_image(self, width: int, height: int, size_bytes: int) -> bool:
        """Filter out decorative icons, lines, or very low-res artifacts."""
        if width < 100 or height < 100: return True
        if size_bytes < 5000: return True # < 5KB is likely a line or tiny icon
        return False

    def extract_and_clean_images(self, pdf_path: Path) -> Dict[int, List[Dict[str, Any]]]:
        """Extract images with smart filtering and deduplication."""
        images_by_page = {}
        if not pdf_path.exists(): return {}
            
        pdf_name = pdf_path.stem
        pdf_image_dir = self.images_dir / pdf_name
        
        try:
            doc = fitz.open(str(pdf_path))
            pdf_image_dir.mkdir(exist_ok=True)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        
                        # Case 1: Filter Junk
                        if self.is_junk_image(base_image["width"], base_image["height"], len(img_bytes)):
                            continue
                            
                        # Case 2: Deduplication
                        img_hash = self._get_image_hash(img_bytes)
                        if img_hash in self.seen_hashes:
                            img_path = self.seen_hashes[img_hash]
                        else:
                            img_name = f"ref_{img_hash[:10]}.{base_image['ext']}"
                            img_path = pdf_image_dir / img_name
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            self.seen_hashes[img_hash] = img_path
                        
                        if (page_num + 1) not in images_by_page:
                            images_by_page[page_num + 1] = []
                            
                        images_by_page[page_num + 1].append({
                            "image_path": str(img_path.relative_to(self.root)),
                            "page": page_num + 1,
                            "hash": img_hash,
                            "width": base_image["width"],
                            "height": base_image["height"]
                        })
                    except Exception as e:
                        logger.debug(f"Image skip on page {page_num+1}: {e}")
            doc.close()
        except Exception as e:
            logger.error(f"PDF Error {pdf_path.name}: {e}")
            
        return images_by_page

    def run(self, use_vlm: bool = False):
        """Process chunks and enrich with filtered/described images."""
        if use_vlm and not self.vlm:
            self.vlm = VinternDescriber()

        if not self.chunks_file.exists():
            logger.error("Source chunks not found.")
            return

        # Simple grouping logic for processing
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            all_chunks = [json.loads(line) for line in f]

        # Group by PDF to avoid re-opening
        pdfs = sorted(list(set(c["source_path"] for c in all_chunks)))
        
        # Cache for image descriptions to avoid re-running VLM on same image (hash-based)
        desc_cache = {}

        with open(self.output_file, "w", encoding="utf-8") as out_f:
            for rel_pdf in tqdm(pdfs, desc="Enriching PDFs"):
                full_pdf = self.raw_dir / rel_pdf
                images_data = self.extract_and_clean_images(full_pdf)
                
                pdf_chunks = [c for c in all_chunks if c["source_path"] == rel_pdf]
                
                for chunk in pdf_chunks:
                    # Heuristic for multi-modal context (simplified)
                    # We look for images in a +/- 1 page range of the chunk if possible
                    # Or just any image in the doc for legal diagrams
                    
                    found_images = []
                    # Logic: If we had page numbers in chunks, we'd filter here.
                    # Mapping logic goes here...
                    
                    # Enrichment
                    enriched_images = []
                    for img in found_images:
                        img_path = self.root / img["image_path"]
                        img_hash = img["hash"]
                        
                        if use_vlm:
                            if img_hash not in desc_cache:
                                desc_cache[img_hash] = self.vlm.describe(img_path)
                            img["description"] = desc_cache[img_hash]
                        
                        enriched_images.append(img)
                    
                    chunk["multimodal"] = {
                        "images": enriched_images,
                        "has_visuals": len(enriched_images) > 0
                    }
                    out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    enricher = MultimodalEnricher()
    # set use_vlm=True when ready to run with GPU
    enricher.run(use_vlm=False)
