import sqlite3
import base64
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from PIL import Image
import io
import hashlib


class ImageDatabase:
    """SQLite database for storing images and metadata"""
    
    def __init__(self, db_path: str = "images.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                file_hash TEXT UNIQUE NOT NULL,
                original_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_extension TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                image_data BLOB NOT NULL,
                thumbnail_data BLOB,
                root_folder TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON images(file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_root_folder ON images(root_folder)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relative_path ON images(relative_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON images(filename)')
        
        conn.commit()
        conn.close()
    
    def _calculate_file_hash(self, image_data: bytes) -> str:
        """Calculate SHA-256 hash of image data"""
        return hashlib.sha256(image_data).hexdigest()
    
    def _create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (200, 200)) -> bytes:
        """Create a thumbnail of the image"""
        # Create a copy to avoid modifying original
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        # Save as JPEG for thumbnails to reduce size
        if thumbnail.mode in ('RGBA', 'LA', 'P'):
            thumbnail = thumbnail.convert('RGB')
        thumbnail.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
        return img_byte_arr.getvalue()
    
    def store_image(self, image_path: Path, root_folder: Path) -> Optional[str]:
        """
        Store an image in the database
        Returns the image ID if successful, None if failed
        """
        try:
            # Load the image
            with Image.open(image_path) as image:
                # Convert to RGB if needed
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                
                # Get image data as bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
                image_data = img_byte_arr.getvalue()
                
                # Calculate file hash
                file_hash = self._calculate_file_hash(image_data)
                
                # Create thumbnail
                thumbnail_data = self._create_thumbnail(image)
                
                # Calculate relative path
                relative_path = str(image_path.relative_to(root_folder))
                
                # Prepare metadata
                image_id = str(uuid.uuid4())
                filename = image_path.name
                file_extension = image_path.suffix.lower()
                file_size = len(image_data)
                width, height = image.size
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if image already exists (by hash)
                cursor.execute('SELECT id FROM images WHERE file_hash = ?', (file_hash,))
                existing = cursor.fetchone()
                
                if existing:
                    print(f"Image already exists in database: {filename}")
                    conn.close()
                    return existing[0]
                
                # Insert new image
                cursor.execute('''
                    INSERT INTO images (
                        id, file_hash, original_path, filename, file_extension,
                        file_size, width, height, image_data, thumbnail_data,
                        root_folder, relative_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id, file_hash, str(image_path.absolute()), filename,
                    file_extension, file_size, width, height, image_data,
                    thumbnail_data, str(root_folder.absolute()), relative_path
                ))
                
                conn.commit()
                conn.close()
                
                print(f"Stored image in database: {filename} (ID: {image_id})")
                return image_id
                
        except Exception as e:
            print(f"Error storing image {image_path}: {e}")
            return None
    
    def get_image(self, image_id: str) -> Optional[Dict]:
        """Get an image by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, file_extension, file_size, width, height,
                   image_data, root_folder, relative_path, created_at
            FROM images WHERE id = ?
        ''', (image_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'filename': result[1],
                'file_extension': result[2],
                'file_size': result[3],
                'width': result[4],
                'height': result[5],
                'image_data': result[6],
                'root_folder': result[7],
                'relative_path': result[8],
                'created_at': result[9]
            }
        return None
    
    def get_thumbnail(self, image_id: str) -> Optional[bytes]:
        """Get thumbnail data for an image"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT thumbnail_data FROM images WHERE id = ?', (image_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_images_by_folder(self, root_folder: str) -> List[Dict]:
        """Get all images from a specific folder"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, file_extension, file_size, width, height,
                   root_folder, relative_path, created_at
            FROM images WHERE root_folder = ?
            ORDER BY created_at DESC
        ''', (root_folder,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'filename': row[1],
                'file_extension': row[2],
                'file_size': row[3],
                'width': row[4],
                'height': row[5],
                'root_folder': row[6],
                'relative_path': row[7],
                'created_at': row[8]
            }
            for row in results
        ]
    
    def get_all_images(self) -> List[Dict]:
        """Get all images from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, file_extension, file_size, width, height,
                   root_folder, relative_path, created_at
            FROM images
            ORDER BY created_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'filename': row[1],
                'file_extension': row[2],
                'file_size': row[3],
                'width': row[4],
                'height': row[5],
                'root_folder': row[6],
                'relative_path': row[7],
                'created_at': row[8]
            }
            for row in results
        ]
    
    def delete_image(self, image_id: str) -> bool:
        """Delete an image from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM images WHERE id = ?', (image_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def delete_images_by_folder(self, root_folder: str) -> int:
        """Delete all images from a specific folder"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM images WHERE root_folder = ?', (root_folder,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def image_exists_by_path(self, relative_path: str, root_folder: str) -> Optional[str]:
        """Check if an image exists by its path, return image ID if exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id FROM images 
            WHERE relative_path = ? AND root_folder = ?
        ''', (relative_path, root_folder))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total images
        cursor.execute('SELECT COUNT(*) FROM images')
        total_images = cursor.fetchone()[0]
        
        # Total size
        cursor.execute('SELECT SUM(file_size) FROM images')
        total_size = cursor.fetchone()[0] or 0
        
        # Images by folder
        cursor.execute('SELECT root_folder, COUNT(*) FROM images GROUP BY root_folder')
        folders = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_images': total_images,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'folders': {folder: count for folder, count in folders}
        }
