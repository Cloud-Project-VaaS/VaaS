const socket = io();

let currentUser = null;
let currentRecipient = null;
let chatHistory = {}; // Store chat messages for each user

document.getElementById("joinBtn").onclick = () => {
  currentUser = document.getElementById("usernameInput").value;
  if (!currentUser) return;
  socket.emit("register", currentUser);

  document.getElementById("loginScreen").style.display = "none";
  document.getElementById("chatScreen").style.display = "block";
};

socket.on("userList", (users) => {
  const usersList = document.getElementById("userList");
  usersList.innerHTML = "";
  users.forEach((user) => {
    if (user !== currentUser) {
      let li = document.createElement("li");
      li.textContent = user;
      li.onclick = () => {
        switchToUser(user);
      };
      usersList.appendChild(li);
    }
  });
});

function switchToUser(user) {
  currentRecipient = user;
  document.getElementById("chatWith").textContent = `Chat with ${user}`;
  
  // Initialize chat history for this user if it doesn't exist
  if (!chatHistory[user]) {
    chatHistory[user] = [];
  }
  
  // Clear chat window and load history for this user
  const chatWindow = document.getElementById("chatWindow");
  chatWindow.innerHTML = "";
  
  // Display all previous messages with this user
  chatHistory[user].forEach(msg => {
    const div = document.createElement("div");
    div.textContent = msg;
    chatWindow.appendChild(div);
  });
  
  // Auto-scroll to bottom
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

document.getElementById("sendBtn").onclick = () => {
  const message = document.getElementById("messageInput").value;
  if (currentRecipient && message) {
    socket.emit("privateMessage", { sender: currentUser, recipient: currentRecipient, message });
    
    // Store the message in chat history
    const messageText = `You -> ${currentRecipient}: ${message}`;
    if (!chatHistory[currentRecipient]) {
      chatHistory[currentRecipient] = [];
    }
    chatHistory[currentRecipient].push(messageText);
    
    // Add to current chat window
    addMessage(messageText);
    document.getElementById("messageInput").value = "";
  }
};

// Allow sending message with Enter key
document.getElementById("messageInput").addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    document.getElementById("sendBtn").click();
  }
});

socket.on("privateMessage", ({ sender, message }) => {
  const messageText = `${sender}: ${message}`;
  
  // Store the received message in chat history
  if (!chatHistory[sender]) {
    chatHistory[sender] = [];
  }
  chatHistory[sender].push(messageText);
  
  // Only show the message if we're currently chatting with this sender
  if (currentRecipient === sender) {
    addMessage(messageText);
  }
});

function addMessage(msg) {
  const chatWindow = document.getElementById("chatWindow");
  const div = document.createElement("div");
  div.textContent = msg;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight; // Auto-scroll to bottom
}
