import React, { useState, useRef, useEffect } from "react";
import "./ChatWidget.css";

const SUGGESTED_QUESTIONS = [
  "What does a high Sensory score mean?",
  "How can I help my child with Social Relationship?",
  "What are the next steps after this screening?",
  "Explain the ISAA behavior domain."
];

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { text: "Hello! I am your Clinical Support Assistant. I can explain your child's scores or answer questions about ASD. How can I help you today?", sender: "bot" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const sendMessage = async (overrideText = null) => {
    const textToSend = overrideText || input;
    if (!textToSend.trim()) return;

    const userMessage = { text: textToSend, sender: "user" };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: textToSend })
      });

      const data = await response.json();
      const botMessage = { text: data.response || "I'm having trouble right now. Please try again.", sender: "bot" };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = { text: "Connection error. Please check if the server is running.", sender: "bot" };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-widget">
      {/* Floating Action Button */}
      <button 
        className={`chat-fab ${isOpen ? "open" : ""}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle chat"
      >
        {isOpen ? "✕" : "💬"}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="chat-window fade-in">
          <div className="chat-header">
            <div className="header-icon">🏥</div>
            <div className="header-text">
              <h3>Clinical Support</h3>
              <p>Specialized ASD Assistant</p>
            </div>
          </div>

          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.sender}`}>
                <div className="message-bubble">{msg.text}</div>
              </div>
            ))}
            {loading && (
              <div className="message bot">
                <div className="message-bubble typing">
                  <span></span><span></span><span></span>
                </div>
              </div>
            )}
            {!loading && messages.length < 3 && (
              <div className="suggestions-container">
                <p className="suggestion-label">Suggested Questions:</p>
                {SUGGESTED_QUESTIONS.map((q, i) => (
                  <button key={i} className="suggestion-chip" onClick={() => sendMessage(q)}>
                    {q}
                  </button>
                ))}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about domains, scores, or next steps..."
            />
            <button className="send-btn" onClick={() => sendMessage()} disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
