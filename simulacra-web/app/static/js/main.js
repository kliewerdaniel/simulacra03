// Main JavaScript for Simulacra Web Interface

    // Initialize global elements when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Dark mode toggle functionality
    initializeDarkMode();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Check for auth token and handle navigation visibility
    checkAuthentication();
    
    // Initialize service worker for offline access
    initializeServiceWorker();
    
    // Add auth token to all page navigation links for protected routes
    addAuthToNavLinks();
});

/**
 * Add auth token to protected route links
 * This ensures routes that require authentication receive the token
 */
function addAuthToNavLinks() {
    // Protected routes that need authentication
    const protectedRoutes = [
        '/dashboard',
        '/document-analysis',
        '/persona-generation',
        '/style-replication'
    ];
    
    // Get all links in the page
    const links = document.querySelectorAll('a');
    links.forEach(link => {
        const href = link.getAttribute('href');
        if (href && protectedRoutes.some(route => href.startsWith(route))) {
            // This is a link to a protected route
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Get auth token
                const token = localStorage.getItem('auth_token');
                if (!token) {
                    // Redirect to login if no token
                    window.location.href = '/auth/login';
                    return;
                }
                
                // Set the token as a cookie if not already set
                if (!document.cookie.includes('auth_token=')) {
                    document.cookie = `auth_token=${token}; path=/; max-age=${30*60}`;
                }
                
                // Create a form to submit with the token
                const form = document.createElement('form');
                form.method = 'GET'; // Changed from POST to GET which is what server expects
                form.action = href;
                form.style.display = 'none';
                
                // Add to body, submit, then remove
                document.body.appendChild(form);
                form.submit();
            });
        }
    });
}

/**
 * Initialize and handle dark mode functionality
 */
function initializeDarkMode() {
    // Check if user has a preferred color scheme
    const prefersDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // Check if user has previously set a preference
    const savedTheme = localStorage.getItem('theme');
    
    // Apply dark mode if preferred or previously set
    if (savedTheme === 'dark' || (prefersDarkMode && savedTheme !== 'light')) {
        document.body.classList.add('dark-mode');
        localStorage.setItem('theme', 'dark');
    }
    
    // Look for dark mode toggle button
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            // Toggle dark mode class
            document.body.classList.toggle('dark-mode');
            
            // Save preference
            const isDarkMode = document.body.classList.contains('dark-mode');
            localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
            
            // Update toggle button text/icon if needed
            if (this.querySelector('i')) {
                const icon = this.querySelector('i');
                if (isDarkMode) {
                    icon.classList.remove('fa-moon');
                    icon.classList.add('fa-sun');
                } else {
                    icon.classList.remove('fa-sun');
                    icon.classList.add('fa-moon');
                }
            }
        });
        
        // Set initial toggle state
        if (darkModeToggle.querySelector('i')) {
            const icon = darkModeToggle.querySelector('i');
            if (document.body.classList.contains('dark-mode')) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            } else {
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            }
        }
    }
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggerList.length > 0) {
        [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
}

/**
 * Check for authentication token and adjust UI
 */
function checkAuthentication() {
    // This is a simplified version - in production, verify with server
    const token = localStorage.getItem('auth_token');
    
    // Adjust UI based on auth status
    if (token) {
        // User is logged in
        document.querySelectorAll('.auth-required').forEach(el => {
            el.style.display = 'block';
        });
        
        document.querySelectorAll('.guest-only').forEach(el => {
            el.style.display = 'none';
        });
    } else {
        // User is not logged in
        document.querySelectorAll('.auth-required').forEach(el => {
            el.style.display = 'none';
        });
        
        document.querySelectorAll('.guest-only').forEach(el => {
            el.style.display = 'block';
        });
    }
}

/**
 * Initialize a service worker for offline access
 */
function initializeServiceWorker() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/static/js/service-worker.js')
            .then(registration => {
                console.log('Service Worker registered with scope:', registration.scope);
            })
            .catch(error => {
                console.error('Service Worker registration failed:', error);
            });
    }
}

/**
 * Format a date string
 * @param {string} dateString - ISO date string
 * @returns {string} Formatted date
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Create and return an error message element
 * @param {string} message - Error message to display
 * @returns {HTMLElement} Error message element
 */
function createErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show';
    errorDiv.role = 'alert';
    errorDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    return errorDiv;
}

/**
 * Create and return a success message element
 * @param {string} message - Success message to display
 * @returns {HTMLElement} Success message element
 */
function createSuccessMessage(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'alert alert-success alert-dismissible fade show';
    successDiv.role = 'alert';
    successDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    return successDiv;
}

/**
 * Show a loading indicator in the specified container
 * @param {string} containerId - ID of the container element
 * @param {string} message - Loading message to display
 */
function showLoading(containerId, message = 'Loading...') {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'text-center my-4 loading-indicator';
    loadingDiv.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2">${message}</p>
    `;
    
    container.innerHTML = '';
    container.appendChild(loadingDiv);
    loadingDiv.style.display = 'block';
}

/**
 * Hide loading indicator in the specified container
 * @param {string} containerId - ID of the container element
 */
function hideLoading(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const loadingIndicator = container.querySelector('.loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }
}

/**
 * Handle API fetch with error handling and authentication
 * @param {string} url - API URL to fetch
 * @param {Object} options - Fetch options
 * @returns {Promise} Fetch promise
 */
async function apiFetch(url, options = {}) {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
        options.headers = options.headers || {};
        options.headers['Authorization'] = `Bearer ${token}`;
        
        // Set the token as a cookie if not already set
        if (!document.cookie.includes('auth_token=')) {
            document.cookie = `auth_token=${token}; path=/; max-age=${30*60}`;
        }
        
        // Always include cookies in the request
        options.credentials = 'include';
    }
    
    try {
        const response = await fetch(url, options);
        
        // Handle unauthorized response
        if (response.status === 401) {
            // Clear token and redirect to login
            localStorage.removeItem('auth_token');
            window.location.href = '/auth/login';
            return null;
        }
        
        // Handle other errors
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({
                detail: `HTTP error! status: ${response.status}`
            }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        // Check if the response is JSON
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }
        
        return await response.text();
    } catch (error) {
        console.error('API fetch error:', error);
        throw error;
    }
}
