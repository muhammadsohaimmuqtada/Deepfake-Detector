import { useState, useRef, useEffect } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Send, Bot, User, Loader2, MessageSquare, X, Minimize2, Maximize2, ShieldAlert } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';

type Message = {
  id: string;
  role: 'user' | 'model';
  text: string;
};

export function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'model',
      text: "SYSTEM ONLINE. Forensic assistant ready. Query regarding deepfakes, AI generation, or digital manipulation?",
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatRef = useRef<any>(null);

  useEffect(() => {
    if (!chatRef.current) {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || process.env.GEMINI_API_KEY || '' });
      chatRef.current = ai.chats.create({
        model: 'gemini-3.1-pro-preview',
        config: {
          systemInstruction: "You are a highly advanced digital forensics AI assistant. You specialize in deepfake detection, AI-generated media, and digital forensics. Answer questions clearly, concisely, and with a slightly technical/forensic tone.",
        },
      });
    }
  }, []);

  useEffect(() => {
    if (isOpen && !isMinimized) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen, isMinimized]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { id: Date.now().toString(), role: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await chatRef.current.sendMessageStream({ message: userMessage.text });
      
      let modelText = '';
      const modelMessageId = (Date.now() + 1).toString();
      
      setMessages((prev) => [...prev, { id: modelMessageId, role: 'model', text: '' }]);

      for await (const chunk of response) {
        modelText += chunk.text;
        setMessages((prev) => 
          prev.map((msg) => 
            msg.id === modelMessageId ? { ...msg, text: modelText } : msg
          )
        );
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), role: 'model', text: "ERR: Connection to forensic core lost. Please retry." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            onClick={() => setIsOpen(true)}
            className="fixed bottom-6 right-6 p-4 bg-indigo-600 text-white rounded-full shadow-[0_0_20px_rgba(99,102,241,0.4)] hover:bg-indigo-500 hover:shadow-[0_0_30px_rgba(99,102,241,0.6)] transition-all z-50 flex items-center justify-center border border-indigo-400/30"
          >
            <ShieldAlert className="w-6 h-6" />
          </motion.button>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ 
              opacity: 1, 
              y: 0, 
              scale: 1,
              height: isMinimized ? 'auto' : '600px',
            }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className={cn(
              "fixed bottom-6 right-6 w-[400px] max-w-[calc(100vw-3rem)] bg-slate-950 rounded-2xl shadow-2xl border border-slate-800 flex flex-col overflow-hidden z-50",
              isMinimized ? "" : "max-h-[calc(100vh-6rem)]"
            )}
          >
            {/* Header */}
            <div className="p-4 bg-slate-900 border-b border-slate-800 text-white flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2 font-mono text-sm uppercase tracking-wider text-indigo-400">
                <Bot className="w-5 h-5" />
                Forensic Comms
              </div>
              <div className="flex items-center gap-1 text-slate-400">
                <button 
                  onClick={() => setIsMinimized(!isMinimized)}
                  className="p-1.5 hover:bg-slate-800 hover:text-white rounded-md transition-colors"
                >
                  {isMinimized ? <Maximize2 className="w-4 h-4" /> : <Minimize2 className="w-4 h-4" />}
                </button>
                <button 
                  onClick={() => setIsOpen(false)}
                  className="p-1.5 hover:bg-rose-500/20 hover:text-rose-400 rounded-md transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Chat Area */}
            {!isMinimized && (
              <>
                <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-950/50">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={cn(
                        "flex gap-3 max-w-[85%]",
                        msg.role === 'user' ? "ml-auto flex-row-reverse" : ""
                      )}
                    >
                      <div className={cn(
                        "w-8 h-8 rounded-full flex items-center justify-center shrink-0 border",
                        msg.role === 'user' ? "bg-indigo-500/20 border-indigo-500/30 text-indigo-400" : "bg-slate-800 border-slate-700 text-slate-400"
                      )}>
                        {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                      </div>
                      <div className={cn(
                        "p-3 rounded-2xl text-sm font-mono",
                        msg.role === 'user' 
                          ? "bg-indigo-600 text-white rounded-tr-sm" 
                          : "bg-slate-900 border border-slate-800 text-slate-300 rounded-tl-sm shadow-sm"
                      )}>
                        {msg.role === 'user' ? (
                          msg.text
                        ) : (
                          <div className="prose prose-sm prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-black prose-pre:border prose-pre:border-slate-800">
                            <Markdown>{msg.text}</Markdown>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {isLoading && (
                    <div className="flex gap-3 max-w-[85%]">
                      <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 text-slate-400 flex items-center justify-center shrink-0">
                        <Bot className="w-4 h-4" />
                      </div>
                      <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 rounded-tl-sm shadow-sm flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                        <span className="text-sm font-mono text-slate-500">PROCESSING...</span>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <form onSubmit={sendMessage} className="p-4 bg-slate-900 border-t border-slate-800 shrink-0">
                  <div className="relative flex items-center">
                    <input
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Enter query..."
                      className="w-full pl-4 pr-12 py-3 bg-slate-950 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl text-sm font-mono text-slate-200 transition-all outline-none"
                    />
                    <button
                      type="submit"
                      disabled={!input.trim() || isLoading}
                      className="absolute right-2 p-2 text-indigo-400 hover:bg-indigo-500/20 rounded-lg disabled:opacity-50 disabled:hover:bg-transparent transition-colors"
                    >
                      <Send className="w-4 h-4" />
                    </button>
                  </div>
                </form>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
