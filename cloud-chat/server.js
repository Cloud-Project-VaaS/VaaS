const express = require("express");
const app = express();
const http = require("http").createServer(app);
const io = require("socket.io")(http);

app.use(express.static("public"));

let users = {}; // username -> socket.id

io.on("connection", (socket) => {
  console.log("A user connected:", socket.id);

  // Register username
  socket.on("register", (username) => {
    users[username] = socket.id;
    console.log(`${username} registered with id ${socket.id}`);
    io.emit("userList", Object.keys(users)); // send updated user list
  });

  // Private message
  socket.on("privateMessage", ({ sender, recipient, message }) => {
    const recipientSocketId = users[recipient];
    if (recipientSocketId) {
      io.to(recipientSocketId).emit("privateMessage", { sender, message });
    }
  });

  socket.on("disconnect", () => {
    for (let username in users) {
      if (users[username] === socket.id) {
        console.log(`${username} disconnected`);
        delete users[username];
        io.emit("userList", Object.keys(users));
        break;
      }
    }
  });
});

http.listen(5000, "0.0.0.0", () => {
  console.log("Server running at http://0.0.0.0:5000");
});

