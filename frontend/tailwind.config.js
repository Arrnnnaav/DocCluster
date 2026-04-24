/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          base: "#0d1117",
          panel: "#161b22",
          hover: "#1c2128",
        },
        border: {
          DEFAULT: "#30363d",
        },
        text: {
          primary: "#e6edf3",
          secondary: "#8b949e",
        },
        accent: {
          DEFAULT: "#7c3aed",
          green: "#10b981",
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
    },
  },
  plugins: [],
};
