# Real-time Chat Application

A simple real-time chat application built with Node.js, Express, and Socket.IO that allows multiple users to communicate privately with each other.

## Features

- ✅ Real-time messaging using WebSocket connections
- ✅ Private messaging between users
- ✅ User registration and online user list
- ✅ Chat history preservation when switching between users

## Tech Stack

- **Backend**: Node.js, Express.js
- **Real-time Communication**: Socket.IO
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Package Manager**: npm

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chatbot
```

2. Install dependencies:
```bash
npm install
```

3. Start the server:
```bash
node server.js
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Enter your username and click "Join Chat"
2. You'll see a list of online users on the left sidebar
3. Click on any user to start a private conversation
4. Type messages and press Enter or click Send
5. Switch between different users - your chat history will be preserved
6. Messages are delivered in real-time to the selected recipient

## Project Structure

```
chatbot/
├── server.js              # Main server file
├── public/                # Static files served to client
│   ├── index.html         # Main HTML page
│   ├── client.js          # Client-side JavaScript
│   ├── style.css          # Styling
│   └── chat.html          # Additional chat page
├── templates/             # Server-side templates
│   ├── login.html
│   └── chat.html
├── package.json           # Project dependencies
├── package-lock.json      # Dependency lock file
└── README.md             # Project documentation
```

## API Endpoints

- `GET /` - Serves the main chat interface
- `GET /socket.io/socket.io.js` - Socket.IO client library
- WebSocket endpoints:
  - `register` - Register a new user
  - `privateMessage` - Send private message
  - `userList` - Broadcast updated user list
  - `disconnect` - Handle user disconnection


