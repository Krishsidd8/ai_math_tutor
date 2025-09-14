const API_BASE_URL = 'https://aimathtutor-production.up.railway.app';

const darkModeToggle = document.getElementById('darkModeToggle');
const body = document.body;
const imageInput = document.getElementById('imageInput');
const solveBtn = document.createElement('button');
solveBtn.id = "solveBtn";
solveBtn.textContent = "Solve";
solveBtn.style.display = "none";

const chatSection = document.getElementById('chatSection');
const uploadBox = document.getElementById('uploadBox');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');

let uploadedImageURL = null;
let spinnerMsg = null;
let equationMsg = null;

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

function cleanLatex(latex) {
  if (!latex) return "";
  latex = latex.trim();
  return latex
    .replace(/```latex/gi, '')
    .replace(/```/g, '')
    .replace(/^'''|'''$/g, '')
    .replace(/^"""/, '')
    .replace(/"""$/, '')
    .replace(/^['"]|['"]$/g, '')
    .replace(/\blatex\b/gi, '')
    .replace(/^\\+/, '')
    .replace(/\\+$/, '')
    .trim();
}

function renderMathInElement(element) {
  if (window.MathJax) {
    MathJax.typesetPromise([element]).catch(err => {
      console.error('MathJax typeset failed:', err.message);
    });
  }
}

function addBotMessage(htmlContent) {
  const botMsg = document.createElement('div');
  botMsg.className = 'chat-message bot';
  botMsg.innerHTML = htmlContent;
  chatSection.appendChild(botMsg);
  renderMathInElement(botMsg);
  chatSection.scrollTop = chatSection.scrollHeight;
  return botMsg;
}

uploadBox.addEventListener('click', () => imageInput.click());
uploadBox.addEventListener('dragover', (e) => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    imageInput.files = files;
    imageInput.dispatchEvent(new Event('change'));
  }
});

imageInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    alert("Please upload a valid image file.");
    imageInput.value = "";
    return;
  }

  const maxSizeMB = 5;
  if (file.size > maxSizeMB * 1024 * 1024) {
    alert(`File size exceeds ${maxSizeMB}MB. Please upload a smaller image.`);
    imageInput.value = "";
    return;
  }

  imagePreviewContainer.innerHTML = '';
  if (spinnerMsg) spinnerMsg.remove();
  if (equationMsg) equationMsg.remove();
  solveBtn.style.display = "none";

  const reader = new FileReader();
  reader.onload = function(e) {
    uploadedImageURL = e.target.result;
    imagePreviewContainer.innerHTML = `<img src="${uploadedImageURL}" alt="Uploaded Preview" />`;
  };
  reader.readAsDataURL(file);

  spinnerMsg = addBotMessage(`<div class="loading-spinner"></div>`);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE_URL}/predict/`, { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      spinnerMsg.remove();
      addBotMessage(`Error from backend: ${data.error}`);
      return;
    }

    if (data.latex) {
      spinnerMsg.remove();
      const latex = cleanLatex(data.latex);
      equationMsg = addBotMessage(`<strong>Predicted Equation:</strong><br>$$${latex}$$`);

      equationMsg.appendChild(solveBtn);
      solveBtn.style.display = "block";
    } else {
      spinnerMsg.remove();
      addBotMessage("No equation predicted.");
    }

  } catch (err) {
    spinnerMsg.remove();
    addBotMessage(`Failed to get equation preview: ${err.message}`);
  }
});

solveBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) {
    alert("Please upload an image.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  solveBtn.disabled = true;
  solveBtn.textContent = "Solving...";

  try {
    const res = await fetch(`${API_BASE_URL}/solve`, { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      addBotMessage(`Error from backend: ${data.error}`);
      return;
    }

    if (data.latex) {
      const cleanedLatex = cleanLatex(data.latex);
      addBotMessage(`<strong>Predicted Equation:</strong><br>$$${cleanedLatex}$$`);
    }

    if (data.steps && Array.isArray(data.steps)) {
      const stepsHtml = data.steps.map(s => {
        const mathjaxRaw = s.mathjax || s.detail || "";
        const mathjax = cleanLatex(mathjaxRaw);
        return `<li><strong>${s.step || ""}:</strong><br>$$${mathjax}$$</li>`;
      }).join('');
      addBotMessage(`<strong>Step-by-Step Solution:</strong><ol>${stepsHtml}</ol>`);
    }

    if (data.note) {
      addBotMessage(`<em>Note: ${data.note}</em>`);
    }

  } catch (err) {
    addBotMessage(`Failed to solve: ${err.message}`);
  } finally {
    solveBtn.disabled = false;
    solveBtn.textContent = "Solve";
  }
});