import { useState } from "react";
import { FiSend } from "react-icons/fi";

export default function QuestionBox({ onSend }) {
  const [question, setQuestion] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    onSend(question.trim());
    setQuestion("");
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-center w-full max-w-2xl border border-gray-500 rounded-full overflow-hidden mt-6 bg-black/30 backdrop-blur-sm"
    >
      <input
        type="text"
        className="flex-1 px-4 py-3 bg-transparent text-white placeholder-gray-400 focus:outline-none"
        placeholder="When is Layla planning her trip to London?"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />
      <button
        type="submit"
        className="p-3 hover:bg-white/10 transition"
        aria-label="Send"
      >
        <FiSend className="text-white text-xl" />
      </button>
    </form>
  );
}