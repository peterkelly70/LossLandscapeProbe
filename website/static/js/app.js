// Main Alpine.js application
document.addEventListener('alpine:init', () => {
    // Global store for shared state
    Alpine.store('app', {
        // State
        sidebarOpen: false,
        darkMode: false,
        loading: false,
        
        // Toggle sidebar on mobile
        toggleSidebar() {
            this.sidebarOpen = !this.sidebarOpen;
            document.body.style.overflow = this.sidebarOpen ? 'hidden' : '';
        },
        
        // Close sidebar when clicking outside
        closeSidebar(event) {
            if (this.sidebarOpen && !event.target.closest('.sidebar') && !event.target.closest('.navbar-burger')) {
                this.sidebarOpen = false;
                document.body.style.overflow = '';
            }
        },
        
        // Toggle dark mode
        toggleDarkMode() {
            this.darkMode = !this.darkMode;
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
                localStorage.setItem('darkMode', 'true');
            } else {
                document.documentElement.classList.remove('dark');
                localStorage.setItem('darkMode', 'false');
            }
        },
        
        // Initialize theme from localStorage
        initTheme() {
            if (localStorage.getItem('darkMode') === 'true' || 
                (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                this.darkMode = true;
                document.documentElement.classList.add('dark');
            }
        },
        
        // Set loading state
        setLoading(state) {
            this.loading = state;
        }
    });
});

// Initialize tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize popovers
function initPopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Initialize clipboard.js for copy buttons
document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme
    Alpine.store('app').initTheme();
    
    // Initialize tooltips and popovers
    initTooltips();
    initPopovers();
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add fade-in animation to elements with .fade-in class
    const fadeElements = document.querySelectorAll('.fade-in');
    const fadeInObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                fadeInObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });
    
    fadeElements.forEach(element => {
        fadeInObserver.observe(element);
    });
});

// Handle loading states for buttons
document.addEventListener('submit', (e) => {
    const form = e.target.closest('form');
    if (form) {
        const submitButton = form.querySelector('[type="submit"]');
        if (submitButton) {
            submitButton.setAttribute('disabled', 'disabled');
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Processing...';
        }
    }
});

// Handle responsive sidebar
document.addEventListener('alpine:init', () => {
    Alpine.data('sidebar', () => ({
        open: false,
        init() {
            this.$watch('open', value => {
                if (value) {
                    document.body.style.overflow = 'hidden';
                } else {
                    document.body.style.overflow = '';
                }
            });
            
            // Close sidebar when clicking outside
            document.addEventListener('click', (event) => {
                if (this.open && !event.target.closest('.sidebar') && !event.target.closest('.navbar-burger')) {
                    this.open = false;
                }
            });
        },
        toggle() {
            this.open = !this.open;
        }
    }));
});
