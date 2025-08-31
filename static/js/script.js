console.log('script.js loaded');

let currentPath = null;
let folderModal = null;
let selectedFolder = null;
let ws = null;

// Initialize WebSocket connection
function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);

    ws.onopen = function () {
        console.log('WebSocket connected');
    };

    ws.onmessage = function (event) {
        const status = JSON.parse(event.data);
        updateIndexingStatus(status);
    };

    ws.onclose = function () {
        console.log('WebSocket disconnected, attempting to reconnect...');
        setTimeout(connectWebSocket, 1000);
    };

    ws.onerror = function (error) {
        console.error('WebSocket error:', error);
    };
}

// Update indexing progress
function updateIndexingStatus(status) {
    const statusDiv = document.getElementById('indexingStatus');
    const progressBar = statusDiv.querySelector('.progress-bar');
    const details = document.getElementById('indexingDetails');

    if (status.status === 'idle') {
        // Fade out the status div
        statusDiv.style.opacity = '0';
        setTimeout(() => {
            statusDiv.style.display = 'none';
            statusDiv.style.opacity = '1';
        }, 500);
        return;
    }

    // Show and update the status
    statusDiv.style.display = 'block';
    statusDiv.style.opacity = '1';

    // Calculate progress percentage
    const percentage = status.total_files > 0
        ? Math.round((status.processed_files / status.total_files) * 100)
        : 0;

    progressBar.style.width = `${percentage}%`;
    progressBar.setAttribute('aria-valuenow', percentage);

    // Update status text
    let statusText = `Status: ${status.status}`;
    if (status.current_file) {
        statusText += ` | Current file: ${status.current_file}`;
    }
    if (status.total_files > 0) {
        statusText += ` | Progress: ${status.processed_files}/${status.total_files} (${percentage}%)`;
    }
    details.textContent = statusText;
}

// IntersectionObserver for lazy loading images
let imageObserver = null;

function observeLazyLoadImages() {
    const lazyLoadImages = document.querySelectorAll('img.lazy-load');

    if (imageObserver) {
        // Disconnect previous observer if any
        imageObserver.disconnect();
    }

    imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                const fullSrc = img.dataset.src;

                if (fullSrc) {
                    img.src = fullSrc;
                    img.removeAttribute('data-src'); // Remove data-src to prevent re-processing
                    img.classList.remove('lazy-load'); // Remove class to prevent re-observing
                }
                observer.unobserve(img); // Stop observing the image once loaded
            }
        });
    }, {
        rootMargin: '0px 0px 200px 0px' // Load images 200px before they enter viewport
    });

    lazyLoadImages.forEach(img => {
        imageObserver.observe(img);
    });
}

// Initialize folder browser
async function initFolderBrowser() {
    folderModal = new bootstrap.Modal(document.getElementById('folderBrowserModal'));
    await loadFolderContents();
    await loadIndexedFolders();
}

// Open folder browser modal
function openFolderBrowser() {
    selectedFolder = null;
    folderModal.show();
    loadFolderContents();
}

function showDrives(breadcrumb, browser, data) {
    // Windows drives
    breadcrumb.innerHTML = '<li class="breadcrumb-item active">Drives</li>';
    data.drives.forEach(drive => {
        const escapedDrive = drive.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        browser.innerHTML += `
                    <div class="folder-item" onclick="loadFolderContents('${escapedDrive}')">
                        <i class="bi bi-hdd"></i>${drive}
                    </div>
                `;
    });
}

function showFolderContents(breadcrumb, browser, data) {
    // Folder contents
    currentPath = data.current_path;

    // Update breadcrumb
    const pathParts = currentPath.split(/[\\/]/);
    let currentBreadcrumb = '';
    pathParts.forEach((part, index) => {
        if (part) {
            // Check if the path contains backslashes to detect Windows
            const isWindows = currentPath.includes('\\');
            currentBreadcrumb += part + (isWindows ? '\\' : '/');
            const isLast = index === pathParts.length - 1;
            const escapedPath = currentBreadcrumb.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
            breadcrumb.innerHTML += `
                                    <li class="breadcrumb-item ${isLast ? 'active' : ''}">
                                        ${isLast ? part : `<a href="#" onclick="loadFolderContents('${escapedPath}')">${part}</a>`}
                                    </li>
                                `;
        }
    });

    // Add parent directory
    if (data.parent_path) {
        addParentDirectory(browser, data);
    }

    // Add folders and files
    addFolderContents(browser, data);
}

function addParentDirectory(browser, data) {
    const escapedParentPath = data.parent_path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
    browser.innerHTML += `
                            <div class="folder-item" onclick="loadFolderContents('${escapedParentPath}')">
                                <i class="bi bi-arrow-up"></i>..
                            </div>
                        `;
}

function addFolderContents(browser, data) {
    data.contents.forEach(item => {
        const icon = item.type === 'directory' ? 'bi-folder' : 'bi-image';
        const escapedPath = item.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        browser.innerHTML += `
                                <div class="folder-item" onclick="${item.type === 'directory' ? `loadFolderContents('${escapedPath}')` : ''}" ondblclick="${item.type === 'directory' ? `selectFolder('${escapedPath}')` : ''}">
                                    <i class="bi ${icon}"></i>${item.name}
                                </div>
                            `;
    });
}

// Load folder contents
async function loadFolderContents(path = null) {
    try {
        const url = path ? `/browse/${encodeURIComponent(path)}` : '/browse';
        const response = await fetch(url);
        const data = await response.json();

        const browser = document.getElementById('folderBrowser');
        const breadcrumb = document.getElementById('folderBreadcrumb');

        browser.innerHTML = '';
        breadcrumb.innerHTML = '';

        if (data.drives) {
            showDrives(breadcrumb, browser, data);
        } else {
            showFolderContents(breadcrumb, browser, data);
        }
    } catch (error) {
        console.error('Error loading folder contents:', error);
    }
}

// Select folder for indexing
function selectFolder(path) {
    selectedFolder = path;
    addSelectedFolder();
}

// Add selected folder
async function addSelectedFolder() {

    folderModal.hide();

    if (!selectedFolder && currentPath) {
        selectedFolder = currentPath;
    }

    if (selectedFolder) {
        try {
            const encodedPath = encodeURIComponent(selectedFolder);
            const response = await fetch(`/folders?folder_path=${encodedPath}`, {
                method: 'POST'
            });

            if (response.ok) {
                await loadIndexedFolders();
                selectedFolder = null;
            } else {
                const error = await response.json();
                alert(`Error adding folder: ${error.detail || error.message || JSON.stringify(error)}`);
            }
        } catch (error) {
            console.error('Error adding folder:', error);
            alert('Error adding folder. Please try again.');
        }
    }
}

// Load indexed folders
async function loadIndexedFolders() {
    try {
        const response = await fetch('/folders');
        const folders = await response.json();

        const folderList = document.getElementById('folderList');
        folderList.innerHTML = '';

        if (folders.length === 0) {
            folderList.innerHTML = `
                <div class="text-center p-4 text-muted">
                    <i class="bi bi-folder-x fs-2 d-block mb-2"></i>
                    <small>No folders indexed yet</small>
                </div>
            `;
            return;
        }

        folders.forEach(folder => {
            const escapedPath = folder.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
            const folderCard = document.createElement('div');
            folderCard.className = `folder-item-card ${!folder.is_valid ? 'invalid' : ''}`;
            folderCard.innerHTML = `
                <div class="d-flex justify-content-between align-items-start p-3">
                    <div class="flex-grow-1 me-2">
                        <div class="d-flex align-items-center mb-1">
                            <i class="bi bi-folder-fill me-2 ${folder.is_valid ? 'text-primary' : 'text-danger'}"></i>
                            <span class="fw-semibold ${!folder.is_valid ? 'text-danger' : 'text-dark'}" style="font-size: 0.9rem;">
                                ${folder.path.split(/[\\/]/).pop()}
                            </span>
                        </div>
                        <div class="text-muted small" style="word-break: break-all; line-height: 1.3;">
                            ${folder.path}
                        </div>
                        ${!folder.is_valid ? '<small class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i>Path not accessible</small>' : ''}
                    </div>
                    <button class="btn btn-outline-danger btn-sm" onclick="removeFolder('${escapedPath}')" title="Remove folder">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            `;
            folderList.appendChild(folderCard);
        });

        // Load images from all folders
        await loadImages();
    } catch (error) {
        console.error('Error loading folders:', error);
    }
}

// Remove folder
async function removeFolder(path) {
    if (confirm('Are you sure you want to remove this folder?')) {
        try {
            const encodedPath = encodeURIComponent(path).replace(/%5C/g, '\\');
            const response = await fetch(`/folders/${encodedPath}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                await loadIndexedFolders();
            } else {
                const error = await response.text();
                alert(`Error removing folder: ${error}`);
            }
        } catch (error) {
            console.error('Error removing folder:', error);
            alert('Error removing folder. Please try again.');
        }
    }
}

// Load images
async function loadImages(folder = null) {
    try {
        const url = folder ? `/images?folder=${encodeURIComponent(folder)}` : '/images';
        const response = await fetch(url);
        const images = await response.json();

        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = '';

        if (images.length === 0) {
            imageGrid.innerHTML = `
                <div class="col-12">
                    <div class="text-center p-5">
                        <i class="bi bi-images fs-1 text-muted d-block mb-3"></i>
                        <h5 class="text-muted mb-2">No images found</h5>
                        <p class="text-muted">Add some folders to start indexing your images</p>
                    </div>
                </div>
            `;
            return;
        }

        images.forEach(image => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.innerHTML = `
                <div class="image-wrapper">
                    <img class="lazy-load"
                         src="/thumbnail/${image.id}"
                         data-src="/image/${image.id}"
                         alt="${image.filename || image.path}"
                         loading="lazy">
                </div>
                <div class="image-info">
                    <span class="filename" title="${image.filename || image.path}">${image.filename || image.path}</span>
                    <span class="file-size">${formatFileSize(image.file_size)}</span>
                </div>
            `;
            imageGrid.appendChild(card);
        });
        observeLazyLoadImages(); // Initialize IntersectionObserver for new images
    } catch (error) {
        console.error('Error loading images:', error);
        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = '<div class="col-12"><div class="error text-center p-4">Error loading images. Please try again.</div></div>';
    }
}

// Utility function to format file sizes
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Get current folder path
function getCurrentPath() {
    // Return the current path if we're in a folder, otherwise null
    return currentPath;
}

// Search images
async function searchImages(event) {
    event.preventDefault();
    const query = document.getElementById('searchInput').value;
    if (!query) return;

    try {
        // Only include folder parameter if we're inside the folder browser
        const searchUrl = `/search/text?query=${encodeURIComponent(query)}`;
        const response = await fetch(searchUrl);
        const results = await response.json();

        displaySearchResults(results);
    } catch (error) {
        console.error('Error searching images:', error);
        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = `
            <div class="col-12">
                <div class="error text-center p-5">
                    <i class="bi bi-exclamation-triangle fs-1 text-danger d-block mb-3"></i>
                    <h5 class="text-danger mb-2">Search Error</h5>
                    <p class="text-muted">An error occurred while searching. Please try again.</p>
                </div>
            </div>
        `;
    }
}

// Search by image
async function searchByImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const searchUrl = '/search/image';
        const response = await fetch(searchUrl, {
            method: 'POST',
            body: formData
        });
        const results = await response.json();

        displaySearchResults(results);

        // Reset file input
        event.target.value = '';
    } catch (error) {
        console.error('Error searching by image:', error);
        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = `
            <div class="col-12">
                <div class="error text-center p-5">
                    <i class="bi bi-exclamation-triangle fs-1 text-danger d-block mb-3"></i>
                    <h5 class="text-danger mb-2">Image Search Error</h5>
                    <p class="text-muted">An error occurred while processing your image. Please try again.</p>
                </div>
            </div>
        `;
    }
}

// Search by URL
async function searchByUrl(event) {
    event.preventDefault();
    const url = document.getElementById('urlInput').value;
    if (!url) return;

    try {
        // Show loading state
        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = `
            <div class="col-12">
                <div class="loading text-center p-5">
                    <div class="spinner-border text-primary mb-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h5 class="text-primary mb-2">Downloading and analyzing image...</h5>
                    <p class="text-muted">This may take a few moments</p>
                </div>
            </div>
        `;

        const searchUrl = `/search/url?url=${encodeURIComponent(url)}`;
        const response = await fetch(searchUrl);
        const results = await response.json();

        displaySearchResults(results);

        // Clear URL input and hide form
        document.getElementById('urlInput').value = '';
        toggleUrlSearch();
    } catch (error) {
        console.error('Error searching by URL:', error);
        const imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = `
            <div class="col-12">
                <div class="error text-center p-5">
                    <i class="bi bi-exclamation-triangle fs-1 text-danger d-block mb-3"></i>
                    <h5 class="text-danger mb-2">Error processing URL</h5>
                    <p class="text-muted">Please check the URL and try again. Make sure it points to a valid image.</p>
                </div>
            </div>
        `;
    }
}

// Display search results (common function for all search types)
function displaySearchResults(results) {
    const imageGrid = document.getElementById('imageGrid');
    imageGrid.innerHTML = '';

    if (results.length === 0) {
        imageGrid.innerHTML = `
            <div class="col-12">
                <div class="no-results text-center p-5">
                    <i class="bi bi-search fs-1 text-muted d-block mb-3"></i>
                    <h5 class="text-muted mb-2">No similar images found</h5>
                    <p class="text-muted">Try adjusting your search terms or uploading a different image</p>
                </div>
            </div>
        `;
        return;
    }

    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.innerHTML = `
            <div class="image-wrapper">
                <img class="lazy-load"
                     src="/thumbnail/${result.id}"
                     data-src="/image/${result.id}"
                     alt="${result.filename || result.path}"
                     loading="lazy">
                <div class="similarity-score">${result.similarity}%</div>
            </div>
            <div class="image-info">
                <span class="filename" title="${result.filename || result.path}">${result.filename || result.path}</span>
                <span class="file-size">${formatFileSize(result.file_size)}</span>
            </div>
        `;
        imageGrid.appendChild(card);
    });
    observeLazyLoadImages(); // Initialize IntersectionObserver for new images
}

// Toggle URL search form visibility
function toggleUrlSearch() {
    const urlForm = document.getElementById('urlSearchForm');
    const isVisible = urlForm.style.display !== 'none';
    
    if (isVisible) {
        urlForm.style.display = 'none';
        document.getElementById('urlInput').value = '';
    } else {
        urlForm.style.display = 'flex';
        document.getElementById('urlInput').focus();
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    initFolderBrowser();
});