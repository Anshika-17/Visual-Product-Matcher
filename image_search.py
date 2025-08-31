import torch
from PIL import Image
from typing import List, Dict, Optional
from transformers import CLIPProcessor, CLIPModel
from qdrant_singleton import QdrantClientSingleton
from folder_manager import FolderManager
from image_database import ImageDatabase
import httpx
import io

class ImageSearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor with proper device handling
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Load model directly to the target device to avoid meta tensor issues
        if self.device == "cuda":
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch16",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # For CPU, use device_map to avoid meta tensor issues
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch16",
                device_map="cpu"
            )
        
        # Initialize Qdrant client, folder manager and image database
        self.qdrant = QdrantClientSingleton.get_instance()
        self.folder_manager = FolderManager()
        self.image_db = ImageDatabase()
    
    def calculate_similarity_percentage(self, score: float) -> float:
        """Convert cosine similarity score to percentage"""
        # Qdrant returns cosine similarity scores between -1 and 1
        # We want to convert this to a percentage between 0 and 100
        # First normalize to 0-1 range, then convert to percentage
        normalized = (score + 1) / 2
        return normalized * 100

    def filter_results(self, search_results: list, threshold: float = 60) -> List[Dict]:
        """Filter and format search results"""
        results = []
        for scored_point in search_results:
            # Convert cosine similarity to percentage
            similarity = self.calculate_similarity_percentage(scored_point.score)
            
            # Only include results above threshold (60% similarity)
            if similarity >= threshold:
                # Get image data from SQLite database
                image_id = scored_point.payload.get("image_id")
                if image_id:
                    image_data = self.image_db.get_image(image_id)
                    if image_data:
                        results.append({
                            "id": image_id,
                            "path": scored_point.payload["path"],
                            "filename": image_data["filename"],
                            "root_folder": scored_point.payload["root_folder"],
                            "similarity": round(similarity, 1),
                            "file_size": image_data["file_size"],
                            "width": image_data["width"],
                            "height": image_data["height"]
                        })
        
        return results
    
    async def search_by_text(self, query: str, folder_path: Optional[str] = None, k: int = 10) -> List[Dict]:
        """Search images by text query"""
        try:
            print(f"\nSearching for text: '{query}'")
            
            # Get collections to search
            collections_to_search = []
            if folder_path:
                # Search in specific folder's collection
                collection_name = self.folder_manager.get_collection_for_path(folder_path)
                if collection_name:
                    collections_to_search.append(collection_name)
                    print(f"Searching in specific folder collection: {collection_name}")
            else:
                # Search in all collections
                folders = self.folder_manager.get_all_folders()
                print(f"Found {len(folders)} folders")
                for folder in folders:
                    print(f"Folder: {folder['path']}, Valid: {folder['is_valid']}, Collection: {folder.get('collection_name', 'None')}")
                # Include all collections regardless of folder validity since images are in SQLite
                collections_to_search.extend(folder["collection_name"] for folder in folders if folder.get("collection_name"))
            
            print(f"Collections to search: {collections_to_search}")
            
            if not collections_to_search:
                print("No collections available to search")
                return []
            
            # Generate text embedding
            inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embedding = text_features.cpu().numpy().flatten()
            
            # Search in all relevant collections
            all_results = []
            for collection_name in collections_to_search:
                try:
                    # Get more results from each collection when searching multiple collections
                    collection_limit = k * 3 if len(collections_to_search) > 1 else k
                    
                    search_result = self.qdrant.search(
                        collection_name=collection_name,
                        query_vector=text_embedding.tolist(),
                        limit=collection_limit,  # Get more results from each collection
                        offset=0,  # Explicitly set offset
                        score_threshold=0.2  # Corresponds to 60% similarity after normalization
                    )
                    
                    # Filter and format results
                    results = self.filter_results(search_result) # Threshold is now default 60 in filter_results
                    all_results.extend(results)
                    print(f"Found {len(results)} matches in collection {collection_name}")
                except Exception as e:
                    print(f"Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort all results by similarity
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Take top k results
            final_results = all_results[:k]
            print(f"Found {len(final_results)} total relevant matches across {len(collections_to_search)} collections")
            
            return final_results
            
        except Exception as e:
            print(f"Error in text search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def search_by_image(self, image: Image.Image, folder_path: Optional[str] = None, k: int = 10) -> List[Dict]:
        """Search images by similarity to uploaded image"""
        try:
            print(f"\nSearching by image...")
            
            # Get collections to search
            collections_to_search = []
            if folder_path:
                # Search in specific folder's collection
                collection_name = self.folder_manager.get_collection_for_path(folder_path)
                if collection_name:
                    collections_to_search.append(collection_name)
                    print(f"Searching in specific folder collection: {collection_name}")
            else:
                # Search in all collections
                folders = self.folder_manager.get_all_folders()
                print(f"Found {len(folders)} folders")
                for folder in folders:
                    print(f"Folder: {folder['path']}, Valid: {folder['is_valid']}, Collection: {folder.get('collection_name', 'None')}")
                # Include all collections regardless of folder validity since images are in SQLite
                collections_to_search.extend(folder["collection_name"] for folder in folders if folder.get("collection_name"))
            
            print(f"Collections to search: {collections_to_search}")
            
            if not collections_to_search:
                print("No collections available to search")
                return []
            
            # Generate image embedding
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_embedding = image_features.cpu().numpy().flatten()
            
            # Search in all relevant collections
            all_results = []
            for collection_name in collections_to_search:
                try:
                    # Get more results from each collection when searching multiple collections
                    collection_limit = k * 3 if len(collections_to_search) > 1 else k
                    
                    search_result = self.qdrant.search(
                        collection_name=collection_name,
                        query_vector=image_embedding.tolist(),
                        limit=collection_limit,  # Get more results from each collection
                        offset=0,  # Explicitly set offset
                        score_threshold=0.2  # Corresponds to 60% similarity after normalization
                    )
                    
                    # Filter and format results
                    results = self.filter_results(search_result) # Threshold is now default 60 in filter_results
                    all_results.extend(results)
                    print(f"Found {len(results)} matches in collection {collection_name}")
                except Exception as e:
                    print(f"Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort all results by similarity
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Take top k results
            final_results = all_results[:k]
            print(f"Found {len(final_results)} total relevant matches across {len(collections_to_search)} collections")
            
            return final_results
            
        except Exception as e:
            print(f"Error in image search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def download_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Download and return an image from a URL"""
        try:
            print(f"Downloading image from URL: {url}")
            
            # Use httpx for async HTTP requests
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Check if the response is an image
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")
                
                # Load image from response content
                image_bytes = io.BytesIO(response.content)
                image = Image.open(image_bytes)
                
                # Convert to RGB if necessary (for consistency with CLIP)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                print(f"Successfully downloaded image: {image.size}")
                return image
                
        except httpx.TimeoutException:
            print(f"Timeout while downloading image from URL: {url}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"HTTP error {e.response.status_code} while downloading image from URL: {url}")
            return None
        except Exception as e:
            print(f"Error downloading image from URL {url}: {e}")
            return None
    
    async def search_by_url(self, url: str, folder_path: Optional[str] = None, k: int = 10) -> List[Dict]:
        """Search images by downloading and comparing an image from a URL"""
        try:
            print(f"\nSearching by image URL: {url}")
            
            # Download the image from URL
            image = await self.download_image_from_url(url)
            if image is None:
                return []
            
            # Use the existing search_by_image method
            return await self.search_by_image(image, folder_path, k)
            
        except Exception as e:
            print(f"Error in URL search: {e}")
            import traceback
            traceback.print_exc()
            return [] 