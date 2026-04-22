/**
 * Medical AI Platform - Shared JS Utilities
 */

// Toast Notifications
const Toast = {
    show(message, type = 'success') {
        const container = document.querySelector('.toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        container.appendChild(toast);

        // Auto remove after 4 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-10px)';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    },
    success(msg) { this.show(msg, 'success'); },
    error(msg) { this.show(msg, 'error'); }
};

// AJAX Form Helper
async function apiPost(url, formData) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Server error');
        return data;
    } catch (err) {
        Toast.error(err.message);
        throw err;
    }
}

// Global UI Handlers
document.addEventListener('DOMContentLoaded', () => {
    // Handle any global UI initializations here
    console.log('Medical AI Platform Initialized');
});
