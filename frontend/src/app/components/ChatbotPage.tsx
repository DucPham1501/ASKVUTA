import { useState, useRef, useEffect } from "react";
import { Send, Mic, Waves, Sparkles, AlertCircle, BookOpen } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

const API_BASE = "/api";

// Scores là raw inner product từ FAISS IndexFlatIP
function formatScore(score: number): string {
  return score.toFixed(2);
}

// Xoá các artifact markdown và metadata khỏi text nguồn trước khi hiển thị
function cleanSourceText(text: string): string {
  return text
    .replace(/##[^\n]*/g, "")           // xoá tiêu đề markdown ## ...
    .replace(/@\S+/g, "")               // xoá @mention và @Shutterstock
    .replace(/\(Ảnh:[^)]*\)/g, "")      // xoá (Ảnh: Sưu tầm)
    .replace(/\|?\s*Nguồn:[^\n]*/g, "") // xoá | Nguồn: ...
    .replace(/Đọc thêm:[^\n]*/g, "")    // xoá Đọc thêm: ...
    .replace(/Xem thêm:[^\n]*/g, "")    // xoá Xem thêm: ...
    .replace(/\s{2,}/g, " ")            // thu gọn khoảng trắng thừa
    .trim();
}

interface DocumentSource {
  chunk_id: number;
  text: string;
  topic: string;
  source: string;
  score: number;
}

interface Message {
  id: number;
  text: string;
  sender: "user" | "bot";
  timestamp: Date;
  sources?: DocumentSource[];
  isError?: boolean;
}

export function ChatbotPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Xin chào! Tôi là trợ lý AI của Vũng Tàu. Tôi có thể giúp bạn tìm hiểu về du lịch, ẩm thực, văn hóa và nhiều thông tin khác về thành phố biển xinh đẹp này. Bạn muốn biết gì?",
      sender: "bot",
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const quickSuggestions = [
    "Quán ăn ngon ở Vũng Tàu",
    "Địa điểm du lịch nổi tiếng",
    "Đặc sản Vũng Tàu",
    "Khách sạn giá rẻ",
    "Lịch sử tượng Chúa Kitô",
    "Thời tiết Vũng Tàu"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSendMessage = async () => {
    if (!inputText.trim() || isTyping) return;

    const question = inputText.trim();

    const userMessage: Message = {
      id: Date.now(),
      text: question,
      sender: "user",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText("");
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5 }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Lỗi máy chủ (${response.status})`);
      }

      const data = await response.json();

      const botMessage: Message = {
        id: Date.now(),
        text: data.answer,
        sender: "bot",
        timestamp: new Date(),
        sources: data.sources?.length ? data.sources : undefined,
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: Date.now(),
        text: error instanceof Error
          ? `Xin lỗi, đã xảy ra lỗi: ${error.message}`
          : "Xin lỗi, không thể kết nối đến máy chủ. Vui lòng thử lại sau.",
        sender: "bot",
        timestamp: new Date(),
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputText(suggestion);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleSources = (messageId: number) => {
    setExpandedSources(prev => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-4 gap-6 h-[calc(100vh-8rem)]">
          {/* Sidebar with suggestions */}
          <div className="lg:col-span-1 space-y-4">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-cyan-100"
            >
              <h3 className="font-semibold text-sky-900 mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-cyan-500" />
                Gợi ý câu hỏi
              </h3>
              <div className="space-y-2">
                {quickSuggestions.map((suggestion, index) => (
                  <motion.button
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="w-full text-left px-4 py-3 bg-gradient-to-r from-cyan-50 to-blue-50 hover:from-cyan-100 hover:to-blue-100 rounded-xl transition-all text-sm text-sky-700 hover:text-sky-900 border border-cyan-200 hover:border-cyan-300 hover:shadow-md"
                  >
                    {suggestion}
                  </motion.button>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl p-6 shadow-lg text-white"
            >
              <Waves className="w-10 h-10 mb-3 opacity-80" />
              <h4 className="font-semibold mb-2">AskVuta</h4>
              <p className="text-sm text-cyan-100">
                Trợ lý thông minh về thành phố biển Vũng Tàu
              </p>
            </motion.div>
          </div>

          {/* Chat Window */}
          <div className="lg:col-span-3">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-2xl border border-cyan-100 flex flex-col h-full overflow-hidden"
            >
              {/* Chat Header */}
              <div className="bg-gradient-to-r from-cyan-500 to-blue-600 px-6 py-4 flex items-center gap-3 border-b border-cyan-400">
                <div className="bg-white/20 backdrop-blur-sm p-2 rounded-xl">
                  <Waves className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">AskVuta</h3>
                  <p className="text-xs text-cyan-100">Trực tuyến</p>
                </div>
              </div>

              {/* Messages Container */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-b from-sky-50/50 to-white/50">
                <AnimatePresence>
                  {messages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div className={`flex gap-3 max-w-[80%] ${message.sender === "user" ? "flex-row-reverse" : "flex-row"}`}>
                        {/* Avatar */}
                        <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
                          message.sender === "bot"
                            ? message.isError
                              ? "bg-gradient-to-br from-red-400 to-orange-500 shadow-md"
                              : "bg-gradient-to-br from-cyan-500 to-blue-600 shadow-md"
                            : "bg-gradient-to-br from-orange-400 to-red-500 shadow-md"
                        }`}>
                          {message.sender === "bot" ? (
                            message.isError
                              ? <AlertCircle className="w-5 h-5 text-white" />
                              : <Waves className="w-5 h-5 text-white" />
                          ) : (
                            <span className="text-white font-semibold">U</span>
                          )}
                        </div>

                        {/* Message Bubble + Sources */}
                        <div className="flex flex-col gap-1">
                          <div className={`px-4 py-3 rounded-2xl shadow-md ${
                            message.sender === "bot"
                              ? message.isError
                                ? "bg-red-50 border border-red-200"
                                : "bg-white border border-cyan-100"
                              : "bg-gradient-to-r from-cyan-500 to-blue-600 text-white"
                          } ${message.sender === "bot" ? "rounded-tl-sm" : "rounded-tr-sm"}`}>
                            <p className={`whitespace-pre-wrap ${
                              message.sender === "bot"
                                ? message.isError ? "text-red-700" : "text-sky-900"
                                : "text-white"
                            }`}>
                              {message.text}
                            </p>
                          </div>

                          {/* Sources toggle */}
                          {message.sources && message.sources.length > 0 && (
                            <div>
                              <button
                                onClick={() => toggleSources(message.id)}
                                className="flex items-center gap-1 text-xs text-cyan-600 hover:text-cyan-800 transition-colors px-1"
                              >
                                <BookOpen className="w-3 h-3" />
                                {expandedSources.has(message.id) ? "Ẩn nguồn" : `Xem ${message.sources.length} nguồn tham khảo`}
                              </button>

                              <AnimatePresence>
                                {expandedSources.has(message.id) && (
                                  <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: "auto" }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="mt-1 space-y-1 overflow-hidden"
                                  >
                                    {message.sources.map((src) => (
                                      <div
                                        key={src.chunk_id}
                                        className="bg-cyan-50 border border-cyan-100 rounded-xl px-3 py-2 text-xs text-sky-700"
                                      >
                                        <div className="flex items-center justify-between mb-1">
                                          <span className="font-medium text-cyan-700">{src.topic}</span>
                                          <span className="text-cyan-500">điểm {formatScore(src.score)}</span>
                                        </div>
                                        <p className="line-clamp-2">{cleanSourceText(src.text)}</p>
                                      </div>
                                    ))}
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </div>
                          )}

                          <p className={`text-xs text-sky-600 px-2 ${message.sender === "user" ? "text-right" : "text-left"}`}>
                            {message.timestamp.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>

                {/* Typing Indicator */}
                {isTyping && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex justify-start"
                  >
                    <div className="flex gap-3 max-w-[80%]">
                      <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-md">
                        <Waves className="w-5 h-5 text-white" />
                      </div>
                      <div className="bg-white border border-cyan-100 px-4 py-3 rounded-2xl rounded-tl-sm shadow-md">
                        <div className="flex gap-1">
                          <motion.div
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ repeat: Infinity, duration: 1, delay: 0 }}
                            className="w-2 h-2 bg-cyan-500 rounded-full"
                          />
                          <motion.div
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ repeat: Infinity, duration: 1, delay: 0.2 }}
                            className="w-2 h-2 bg-cyan-500 rounded-full"
                          />
                          <motion.div
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ repeat: Infinity, duration: 1, delay: 0.4 }}
                            className="w-2 h-2 bg-cyan-500 rounded-full"
                          />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="border-t border-cyan-100 p-4 bg-white">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Nhập câu hỏi của bạn..."
                    disabled={isTyping}
                    className="flex-1 px-4 py-3 bg-sky-50 border border-cyan-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent text-sky-900 placeholder-sky-400 disabled:opacity-60"
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={!inputText.trim() || isTyping}
                    className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white p-3 rounded-xl hover:from-cyan-600 hover:to-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                  <button className="bg-gradient-to-r from-orange-400 to-red-500 text-white p-3 rounded-xl hover:from-orange-500 hover:to-red-600 transition-all shadow-md hover:shadow-lg">
                    <Mic className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
