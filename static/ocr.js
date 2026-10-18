const API_URL = "/process";
const processButton = document.getElementById('processButton');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');
const transcribedTextDiv = document.getElementById('transcribedText');
const translatedTextDiv = document.getElementById('translatedText');
const statusMessageDiv = document.getElementById('statusMessage');

// --- Utility Functions ---

/**
 * Converts a File object into a Base64 string for API payload.
 * @param {File} file The image file to convert.
 * @returns {Promise<string>} Base64 encoded string.
 */
const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
};

/**
 * Updates the UI to show loading or error status.
 * @param {string} text The message to display.
 * @param {string} type 'loading' or 'error' or 'success'.
 */
const updateStatus = (text, type = 'loading') => {
    if (!statusMessageDiv) return;
    statusMessageDiv.textContent = text;
    statusMessageDiv.classList.remove('hidden', 'bg-red-100', 'text-red-700', 'bg-blue-100', 'text-blue-700', 'bg-green-100', 'text-green-700');
    
    if (type === 'error') {
        statusMessageDiv.classList.add('bg-red-100', 'text-red-700');
    } else if (type === 'success') {
        statusMessageDiv.classList.add('bg-green-100', 'text-green-700');
    } else { // 'loading'
        statusMessageDiv.classList.add('bg-blue-100', 'text-blue-700');
    }
};

const resetUI = () => {
    if (transcribedTextDiv) transcribedTextDiv.innerHTML = 'Waiting for transcription...';
    if (translatedTextDiv) translatedTextDiv.innerHTML = 'Waiting for translation...';
    if (statusMessageDiv) statusMessageDiv.classList.add('hidden');
    if (processButton) {
        processButton.textContent = 'Process Image';
        processButton.disabled = false;
        processButton.classList.remove('bg-gray-400', 'cursor-not-allowed', 'animate-pulse');
        processButton.classList.add('bg-indigo-600', 'hover:bg-indigo-700');
    }
};

// --- Core Logic ---

const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (imagePreview) imagePreview.src = e.target.result;
            if (imagePreviewContainer) imagePreviewContainer.classList.remove('hidden');
            if (processButton) {
                processButton.disabled = false;
                processButton.classList.remove('bg-gray-400', 'cursor-not-allowed');
                processButton.classList.add('bg-indigo-600', 'hover:bg-indigo-700');
            }
            resetUI();
        };
        reader.readAsDataURL(file);
    } else {
        if (imagePreviewContainer) imagePreviewContainer.classList.add('hidden');
        if (imagePreview) imagePreview.src = '';
        if (processButton) {
            processButton.disabled = true;
            processButton.classList.add('bg-gray-400', 'cursor-not-allowed');
            processButton.classList.remove('bg-indigo-600', 'hover:bg-indigo-700');
        }
    }
};

const processImage = async () => {
    const file = imageInput?.files?.[0];
    if (!file) {
        updateStatus('Please select an image file.', 'error');
        return;
    }

    if (processButton) {
        processButton.disabled = true;
        processButton.textContent = 'Processing...';
        processButton.classList.add('animate-pulse');
    }
    if (transcribedTextDiv) transcribedTextDiv.textContent = 'Extracting text from image...';
    if (translatedTextDiv) translatedTextDiv.textContent = 'Deciphering modern English translation...';

    try {
        const base64Image = await fileToBase64(file);
        updateStatus('Deciphering epigraphy & formulating translation...', 'loading');

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64Image,
                mime_type: file.type || 'image/jpeg'
            })
        });

        if (!response.ok) {
            const errJson = await response.json().catch(() => ({}));
            throw new Error(errJson.error || `Processing failed with status ${response.status}`);
        }

        const data = await response.json();
        const transcribedText = (data.text || data.raw_text || '').trim() || 'Could not extract text.';
        const translatedText = (data.translation || data.english_translation || data.meaning || '').trim() || 'Translation unavailable.';

        if (transcribedTextDiv) transcribedTextDiv.textContent = transcribedText;
        if (translatedTextDiv) translatedTextDiv.textContent = translatedText;

        updateStatus('Processing Complete!', 'success');
        setTimeout(() => { if (statusMessageDiv) statusMessageDiv.classList.add('hidden'); }, 3000);
    } catch (error) {
        console.error("Processing Error:", error);
        const errorMessage = "An error occurred during processing. Please check the console for details or try again.";
        updateStatus(errorMessage, 'error');
        transcribedTextDiv.textContent = 'Failed to extract text.';
        translatedTextDiv.textContent = 'Failed to translate.';
    } finally {
        // Reset button state
        processButton.disabled = false;
        processButton.textContent = 'Process Image';
        processButton.classList.remove('animate-pulse');
    }
};

// --- Event Listeners ---
imageInput.addEventListener('change', handleImageUpload);
processButton.addEventListener('click', processImage);

// Initial setup for the button
document.addEventListener('DOMContentLoaded', () => {
    // Ensure the button state is correctly set on load
    handleImageUpload({ target: imageInput }); 
});