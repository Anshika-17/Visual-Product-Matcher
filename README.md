# Visual Product Search 🔍

An intelligent visual search engine that revolutionizes product discovery using state-of-the-art AI technology. This application combines CLIP (Contrastive Language-Image Pre-Training) with Qdrant vector database to enable semantic search across image collections, making it perfect for e-commerce, inventory management, and content discovery.

## 🌟 Key Features

- 🎯 **Multi-Modal Search**: Search using text descriptions, uploaded images, or image URLs
- 🖼️ **Smart Indexing**: Automatically indexes and monitors image folders with real-time updates
- 🔍 **Semantic Understanding**: Uses OpenAI's CLIP model for deep image-text comprehension
- � **Similarity Scoring**: Provides percentage-based similarity scores for accurate results
- ⚡ **Real-time Processing**: WebSocket-powered live progress updates during indexing
- 🎨 **Modern UI**: Clean, responsive interface with advanced search capabilities
- 🌐 **URL Support**: Direct image search from web URLs
- 📱 **Mobile Responsive**: Works seamlessly across all devices

## 🧠 Technical Approach & Solution

### Problem Statement
Traditional image search relies on metadata and filenames, which often fail to capture the actual visual content. Users struggle to find specific products or images without knowing exact file names or having perfect tagging systems.

### Our Solution Architecture

#### 1. **Multi-Modal Embedding Generation**
```
Text Query → CLIP Text Encoder → 512D Vector
Image Input → CLIP Vision Encoder → 512D Vector
URL Image → Download → CLIP Vision Encoder → 512D Vector
```

#### 2. **Vector Similarity Search**
- **Database**: Qdrant cloud vector database for scalable similarity search
- **Indexing**: Real-time folder monitoring with automatic embedding generation
- **Storage**: Hybrid approach - embeddings in Qdrant, metadata in SQLite

#### 3. **Semantic Matching Pipeline**
```
User Input → Feature Extraction → Vector Search → Similarity Ranking → Results
```

### �️ Architecture Components

#### Backend (FastAPI)
- **Image Processing**: PIL + CLIP for feature extraction
- **Vector Operations**: Qdrant client for similarity search
- **File Management**: Automatic folder monitoring and indexing
- **API Endpoints**: RESTful APIs for all search operations

#### Frontend (Modern Web UI)
- **Framework**: Vanilla JavaScript with Bootstrap 5
- **Styling**: Custom CSS with modern design principles
- **Real-time Updates**: WebSocket connections for live progress
- **Responsive Design**: Mobile-first approach

#### Database Layer
- **Vector Storage**: Qdrant cloud for embeddings and similarity search
- **Metadata Storage**: SQLite for image metadata and file information
- **Caching**: Thumbnail generation and caching for performance

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, recommended for performance)
- Qdrant Cloud account (free tier available)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/itsfuad/SnapSeek
cd SnapSeek
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
Create a `.env` file:
```env
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_cluster_url
```

5. **Launch the application**:
```bash
python app.py
```

6. **Access the interface**:
Open http://localhost:8000 in your browser

## 🎯 Usage Guide

### 1. **Index Your Images**
- Click "Add Folder" to select image directories
- Watch real-time indexing progress
- Images are automatically monitored for changes

### 2. **Search Methods**

#### Text Search
```
"red sports car"
"woman wearing blue dress"
"modern kitchen design"
```

#### Image Upload Search
- Click the image icon
- Upload a reference image
- Get visually similar results

#### URL Search
- Click the link icon
- Paste any image URL
- Find similar images in your collection

### 3. **Results & Insights**
- Similarity percentages for each match
- High-resolution image previews
- Metadata and file information

## 🏭 Production Deployment

### Recommended Platforms

#### 1. **Railway (Recommended)**
- **Why**: Best for AI/ML applications with generous free tier
- **Resources**: 512MB RAM, 1GB storage
- **Benefits**: No sleep mode, automatic GitHub deployments

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. **Render**
- **Resources**: 512MB RAM, 1GB storage
- **Benefits**: Free SSL, auto-deploy, no cold starts

#### 3. **Fly.io**
- **Resources**: 256MB RAM, 3GB storage volume
- **Benefits**: Global edge deployment, persistent volumes

### Environment Variables for Production
```env
QDRANT_API_KEY=your_production_key
QDRANT_URL=your_production_cluster
PORT=8000
DATA_DIR=/app/data
```

## 🛠️ Development & Testing

### Project Structure
```
SnapSeek/
├── app.py                 # FastAPI application
├── image_indexer.py       # Image processing and indexing
├── image_search.py        # Search logic and CLIP integration
├── image_database.py      # Database operations
├── folder_manager.py      # Folder monitoring and management
├── qdrant_singleton.py    # Qdrant client management
├── requirements.txt       # Dependencies
├── .env                   # Environment configuration
├── templates/
│   └── index.html        # Main UI template
├── static/
│   ├── js/
│   │   └── script.js     # Frontend JavaScript
│   └── image.png         # Application icon
├── config/
│   └── folders.json      # Folder configuration
└── tests/
    └── test_*.py         # Test files
```

### Running Tests
```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-test.txt

# Run with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 Performance Optimization

### Model Selection
```python
# For production (smaller, faster)
MODEL_NAME = "openai/clip-vit-base-patch16"

# For development (balance)
MODEL_NAME = "openai/clip-vit-base-patch32"
```

### Hardware Recommendations
- **CPU**: 4+ cores for concurrent processing
- **RAM**: 8GB+ for model loading and image processing
- **Storage**: SSD recommended for faster I/O
- **GPU**: Optional, CUDA-compatible for faster inference

### Scaling Considerations
- **Batch Processing**: Process multiple images simultaneously
- **Caching**: Implement Redis for frequent queries
- **Load Balancing**: Use multiple instances for high traffic
- **Database Sharding**: Split collections by categories

## 🐛 Troubleshooting

### Common Issues

#### 1. **Model Loading Errors**
```bash
# Clear cache and reinstall
pip uninstall torch torchvision transformers
pip install torch torchvision transformers --no-cache-dir
```

#### 2. **Qdrant Connection Issues**
- Verify API key and URL in `.env`
- Check network connectivity
- Ensure Qdrant cluster is active

#### 3. **Memory Issues**
- Reduce batch size in processing
- Use CPU-only mode: `device="cpu"`
- Close unused applications

#### 4. **Slow Performance**
- Enable GPU acceleration
- Optimize image sizes
- Implement result caching

### Performance Monitoring
```python
# Add logging for performance tracking
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Time search operations
start_time = time.time()
results = await searcher.search_by_text(query)
logger.info(f"Search completed in {time.time() - start_time:.2f}s")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Create a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write tests for new features

## 📊 Use Cases & Applications

### E-commerce
- Product recommendation systems
- Visual search for online stores
- Inventory management
- Duplicate product detection

### Content Management
- Digital asset organization
- Stock photo searching
- Brand consistency checking
- Content moderation

### Research & Education
- Academic image databases
- Scientific data analysis
- Historical archive searches
- Educational content discovery

## 🔮 Future Enhancements

- [ ] **Multi-language Support**: Extend text search to multiple languages
- [ ] **Advanced Filters**: Add size, color, and metadata filters
- [ ] **Batch Operations**: Upload and search multiple images at once
- [ ] **API Integration**: RESTful API for external applications
- [ ] **Machine Learning**: Custom fine-tuned models for specific domains
- [ ] **Analytics Dashboard**: Search metrics and usage statistics
- [ ] **Mobile App**: Native mobile applications
- [ ] **Cloud Storage**: Integration with AWS S3, Google Drive, etc.

## 📄 License

This project is licensed under the Mozilla Public License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI**: For the CLIP model and research
- **Qdrant**: For the excellent vector database
- **FastAPI**: For the modern web framework
- **Transformers**: For the model implementation
- **Bootstrap**: For the UI components

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/itsfuad/SnapSeek/issues)
- **Discussions**: [GitHub Discussions](https://github.com/itsfuad/SnapSeek/discussions)
- **Documentation**: [Wiki](https://github.com/itsfuad/SnapSeek/wiki)

---

**Made with ❤️ by [itsfuad](https://github.com/itsfuad)**

*Revolutionizing visual search with AI technology* 
