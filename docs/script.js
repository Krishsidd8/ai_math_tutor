const API_BASE_URL = 'https://aimathtutor-production.up.railway.app';

const darkModeToggle = document.getElementById('darkModeToggle');
const body = document.body;
const imageInput = document.getElementById('imageInput');
const solveBtn = document.getElementById('solveBtn');
const chatSection = document.getElementById('chatSection');
const uploadBox = document.getElementById('uploadBox');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');

let uploadedImageURL = null;

function updateToggleText() {
  darkModeToggle.textContent = body.classList.contains('dark-mode')
    ? '☀️ Light Mode'
    : '🌙 Dark Mode';
}

darkModeToggle.addEventListener('click', () => {
  body.classList.toggle('dark-mode');
  updateToggleText();
});

updateToggleText();

uploadBox.addEventListener('click', () => {
  imageInput.click();
});

uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
  uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    imageInput.files = files;
    const event = new Event('change');
    imageInput.dispatchEvent(event);
  }
});

imageInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    uploadedImageURL = e.target.result;
    imagePreviewContainer.innerHTML = `<img src="${uploadedImageURL}" alt="Uploaded Preview" />`;
  };
  reader.readAsDataURL(file);
});

solveBtn.addEventListener('click', () => {
  if (!imageInput.files[0]) {
    alert("Please upload an image.");
    return;
  }

  const file = imageInput.files[0];
  const formData = new FormData();
  formData.append("image", file);

  fetch(`${API_BASE_URL}/solve`, {
    method: "POST",
    body: formData,
  })
  .then(res => res.json())
  .then(data => {
    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    const botMsg = document.createElement('div');
    botMsg.className = 'chat-message bot';
    botMsg.innerHTML = `
      <strong>Predicted LaTeX:</strong> ${data.latex}<br/>
      <strong>Step-by-Step Solution:</strong>
      <ol>${data.steps.map(s => `<li>${s.step}<br/><code>${s.symbolic}</code></li>`).join('')}</ol>
    `;
    chatSection.appendChild(botMsg);
    chatSection.scrollTop = chatSection.scrollHeight;
  })
  .catch(err => alert("Failed to solve: " + err));
});