document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const spinner = document.getElementById('loadingSpinner');

    form.addEventListener('submit', function() {
        spinner.style.display = 'block';
    });
});
