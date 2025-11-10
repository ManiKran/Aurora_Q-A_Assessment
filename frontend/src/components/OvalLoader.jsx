import { motion } from "framer-motion";

export default function OvalLoader() {
  const ovalVariants = {
    expand: {
      scale: [0.6, 1, 1.2],
      opacity: [0.9, 0.6, 0],
      transition: {
        duration: 2.8,
        repeat: Infinity,
        ease: "easeInOut",
        staggerChildren: 0.5,
      },
    },
  };

  return (
    <div className="flex items-center justify-center h-screen w-full bg-gradient-to-br from-black via-[#2b190a] to-[#ffb347] overflow-hidden">
      <div className="relative flex items-center justify-center">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            variants={ovalVariants}
            animate="expand"
            className="absolute border-8 border-white rounded-full"
            style={{
              width: `${260 - i * 40}px`,
              height: `${160 - i * 25}px`,
              borderRadius: "50% / 40%",
              opacity: 0.7 - i * 0.15,
            }}
          />
        ))}
      </div>
    </div>
  );
}