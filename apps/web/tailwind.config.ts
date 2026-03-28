import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        board: {
          bg: "#DCB35C",
          line: "#8B6914",
        },
        stone: {
          black: "#1a1a1a",
          white: "#f5f5f5",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
