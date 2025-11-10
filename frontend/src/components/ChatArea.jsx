import TypingText from "./TypingText";
import { motion } from "framer-motion";

export default function ChatArea({ messages, isTyping }) {
  return (
    <div className="w-full max-w-2xl mt-6 space-y-4 overflow-y-auto h-[50vh] px-2">
      {messages.map((msg, idx) => (
        <motion.div
          key={idx}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
        >
          <div
            className={`px-4 py-2 rounded-2xl max-w-[80%] text-sm md:text-base ${
              msg.role === "user"
                ? "bg-white text-black"
                : "bg-[#2b190a]/70 text-white border border-white/10"
            }`}
          >
            {/* If assistant and still thinking, show animated dots */}
            {msg.role === "assistant" && isTyping && idx === messages.length - 1 ? (
              <motion.span
                initial={{ opacity: 0.3 }}
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                Thinking<span className="animate-pulse">...</span>
              </motion.span>
            ) : msg.role === "assistant" && !isTyping && idx === messages.length - 1 ? (
              <TypingText text={msg.content} />
            ) : (
              msg.content
            )}
          </div>
        </motion.div>
      ))}
    </div>
  );
}