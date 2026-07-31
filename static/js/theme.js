(function() {
    // Function to apply theme
    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    }

    // Function to toggle theme
    function toggleTheme() {
        var currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        var nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setTheme(nextTheme);
    }

    // Attach click listeners on DOM ready
    function initThemeToggle() {
        var toggleBtns = document.querySelectorAll('#theme-toggle, .theme-toggle-btn');
        toggleBtns.forEach(function(btn) {
            btn.removeEventListener('click', toggleTheme);
            btn.addEventListener('click', toggleTheme);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initThemeToggle);
    } else {
        initThemeToggle();
    }
})();
