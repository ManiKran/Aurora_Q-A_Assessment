import { useState } from "react";
import { motion } from "framer-motion";
import QuestionBox from "./QuestionBox";
import ChatArea from "./ChatArea";

export default function Landing() {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async (question) => {
    const newMessages = [
      { role: "user", content: question },
      { role: "assistant", content: "Thinking..." },
    ];
    setMessages(newMessages);
    setIsTyping(true);

    try {
      const res = await fetch(
        `https://auroraq-aassessment-production.up.railway.app/ask?question=${encodeURIComponent(
          question
        )}`
      );
      const data = await res.json();
      const answer = data.answer || "No answer found.";

      setMessages([
        { role: "user", content: question },
        { role: "assistant", content: answer },
      ]);
    } catch (err) {
      setMessages([
        { role: "user", content: question },
        {
          role: "assistant",
          content: "Error fetching answer. Please try again.",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <motion.div
      className="relative flex flex-col items-center justify-center h-screen text-center overflow-hidden px-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1.2 }}
    >
      {/* ✨ Animated Gradient Background */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-br from-black via-[#2b190a] to-[#ffb347] z-[-1]"
        animate={{
          backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
        }}
        transition={{
          duration: 12,
          ease: "easeInOut",
          repeat: Infinity,
        }}
        style={{
          backgroundSize: "200% 200%",
        }}
      />

      {/* Header */}
      <motion.h1
        className="text-3xl md:text-4xl font-semibold mb-4 tracking-wide"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.8 }}
      >
        Ask questions about our members.
      </motion.h1>

      {/* Input + Chat */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.8 }}
        className="w-full flex flex-col items-center"
      >
        <QuestionBox onSend={handleSend} />
        <ChatArea messages={messages} isTyping={isTyping} />
      </motion.div>

      {/* Glowing Aurora Title */}
      <motion.h1
        className="absolute bottom-0 text-[20vw] font-serif text-white/90 opacity-90 select-none pointer-events-none bg-gradient-to-r from-white via-amber-400 to-white bg-clip-text text-transparent"
        animate={{
          backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
        }}
        transition={{
          duration: 6,
          ease: "easeInOut",
          repeat: Infinity,
        }}
        style={{
          backgroundSize: "200% 200%",
          filter: "drop-shadow(0 0 25px rgba(255, 255, 255, 0.25))",
        }}
      >
        AURORA
      </motion.h1>
    </motion.div>
  );
}   