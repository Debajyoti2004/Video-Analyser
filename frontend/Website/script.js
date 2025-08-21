document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selection ---
    const body = document.body;
    const themeToggleBtns = document.querySelectorAll('.theme-toggle-btn');
    const hamburgerMenu = document.getElementById('hamburgerMenu');
    const mobileSidebar = document.getElementById('mobileSidebar');
    const overlay = document.getElementById('overlay');
    
    // Desktop columns and resizers
    const leftCol = document.getElementById('column-left');
    const centerCol = document.getElementById('column-center');
    const rightCol = document.getElementById('column-right');
    const resizerLeft = document.getElementById('resizer-left');
    const resizerRight = document.getElementById('resizer-right');
    
    const fileInput = document.getElementById('fileInput');
    const queryInput = document.getElementById('queryInput');
    const askButton = document.getElementById('askButton');
    const chatMessages = document.getElementById('chatMessages');
    const historyListDesktop = document.getElementById('historyListDesktop');

    // --- Theme Switching ---
    themeToggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            body.classList.toggle('dark-mode');
            body.classList.toggle('light-mode');
        });
    });

    // --- Mobile Sidebar Logic ---
    function syncMobileSidebar() {
        mobileSidebar.innerHTML = ''; // Clear previous content

        // Clone sections from the main desktop view
        const userSection = document.getElementById('user-section-desktop').cloneNode(true);
        const uploadSection = document.getElementById('upload-section-desktop').cloneNode(true);
        const historySection = document.getElementById('history-section-desktop').cloneNode(true);
        
        // Append in the correct order: User -> Upload -> History
        mobileSidebar.appendChild(userSection);
        mobileSidebar.appendChild(uploadSection);
        mobileSidebar.appendChild(historySection);
    }

    hamburgerMenu.addEventListener('click', () => {
        syncMobileSidebar(); // Build the sidebar content on demand
        mobileSidebar.classList.add('open');
        overlay.classList.add('active');
    });

    overlay.addEventListener('click', () => {
        mobileSidebar.classList.remove('open');
        overlay.classList.remove('active');
    });

    // --- File Upload Logic ---
    document.addEventListener('click', function(event) {
        // Use event delegation for buttons that might be in the mobile sidebar
        const target = event.target.closest('.upload-button');
        if (target) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;
        displayFilePreview(file);
        simulateProgress(file);
    });

    function displayFilePreview(file) {
        const fileURL = URL.createObjectURL(file);
        const fileType = file.type.split('/')[0];
        
        document.querySelectorAll('.video-preview').forEach(el => {
             el.style.display = 'none';
             if (fileType === 'video') { el.src = fileURL; el.style.display = 'block'; }
        });
        document.querySelectorAll('.image-preview').forEach(el => {
             el.style.display = 'none';
             if (fileType === 'image') { el.src = fileURL; el.style.display = 'block'; }
        });
        document.querySelectorAll('.upload-placeholder').forEach(el => el.style.display = 'none');
    }
    
    function simulateProgress(file) {
        let progress = 0;
        document.querySelectorAll('.progress-bar').forEach(el => el.style.width = '0%');
        document.querySelectorAll('.progress-text').forEach(el => el.textContent = '0%');
        
        const interval = setInterval(() => {
            progress += 10;
            if (progress > 100) progress = 100;
            document.querySelectorAll('.progress-bar').forEach(el => el.style.width = `${progress}%`);
            document.querySelectorAll('.progress-text').forEach(el => el.textContent = `${progress}%`);
            if (progress >= 100) { clearInterval(interval); }
        }, 200);
    }

    // --- Chat Logic ---
    askButton.addEventListener('click', handleAskQuestion);
    queryInput.addEventListener('keyup', (e) => e.key === 'Enter' && handleAskQuestion());

    async function handleAskQuestion() {
        const query = queryInput.value.trim();
        if (!query) return;

        addMessage(query, 'user');
        updateHistory(query);
        queryInput.value = '';

        toggleLoading(true);
        const answer = await fetch_answer(query);
        addMessage(answer, 'assistant');
        toggleLoading(false);
    }

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        const avatarIcon = sender === 'user' ? 'bi-person-circle' : 'bi-robot';
        messageDiv.innerHTML = `<i class="bi ${avatarIcon} avatar"></i><div class="message-content">${text}</div>`;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function updateHistory(query) {
        const listItem = document.createElement('li');
        listItem.textContent = query;
        historyListDesktop.prepend(listItem);
    }

    function toggleLoading(isLoading) {
        const loader = askButton.querySelector('.loader');
        const askText = askButton.querySelector('.ask-text');
        askButton.disabled = isLoading;
        queryInput.disabled = isLoading;
        loader.style.display = isLoading ? 'block' : 'none';
        askText.style.display = isLoading ? 'none' : 'flex'; // Use flex for centering
    }

    async function fetch_answer(query) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        return `This is a simulated AI response regarding your query about: "${query}".`;
    }

    // --- Column Resizer Logic ---
    function initResizer(resizerEl, leftPanel, rightPanel) {
        let x = 0;
        let leftWidth = 0;

        const onMouseDown = (e) => {
            e.preventDefault();
            x = e.clientX;
            leftWidth = leftPanel.getBoundingClientRect().width;
            
            resizerEl.classList.add('resizing');
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('mouseup', onMouseUp);
        };

        const onMouseMove = (e) => {
            const dx = e.clientX - x;
            const newLeftWidth = leftWidth + dx;
            const totalWidth = leftPanel.offsetWidth + rightPanel.offsetWidth;

            if (newLeftWidth > 250 && newLeftWidth < totalWidth - 250) {
                leftPanel.style.flex = `0 0 ${newLeftWidth}px`;
                // The right panel's width will be calculated automatically by flexbox
            }
        };

        const onMouseUp = () => {
            resizerEl.classList.remove('resizing');
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
        };
        
        resizerEl.addEventListener('mousedown', onMouseDown);
    }

    initResizer(resizerLeft, leftCol, centerCol);
    initResizer(resizerRight, centerCol, rightCol);
});