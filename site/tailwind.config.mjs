/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#F7F7F8",
          100: "#EDEDEF",
          200: "#D6D6DA",
          300: "#B3B3B9",
          400: "#85858D",
          500: "#5C5C64",
          600: "#3D3D44",
          700: "#26262B",
          800: "#16161A",
          900: "#0C0C0F",
          950: "#06060A",
        },
        paper: {
          50: "#FFFFFF",
          100: "#FAFAF7",
          200: "#F4F4EE",
          300: "#E9E9E1",
          400: "#D6D6CB",
        },
        aurora: {
          50: "#EEF1FF",
          100: "#DDE3FF",
          200: "#B8C5FF",
          300: "#8FA1FF",
          400: "#6C82FF",
          500: "#4F66F5",
          600: "#3B4EE0",
          700: "#2E3DB8",
          800: "#222C8C",
          900: "#171E66",
        },
        accent: {
          400: "#A78BFA",
          500: "#8B5CF6",
          600: "#7C3AED",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        display: [
          "Inter Tight",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },
      letterSpacing: {
        tightest: "-0.045em",
        tighter: "-0.035em",
        tightx: "-0.025em",
      },
      fontSize: {
        "display-2xl": [
          "clamp(3.75rem, 7.2vw + 0.5rem, 7.5rem)",
          { lineHeight: "0.95", letterSpacing: "-0.05em", fontWeight: "600" },
        ],
        "display-xl": [
          "clamp(3rem, 5.6vw + 0.4rem, 5.5rem)",
          { lineHeight: "1.0", letterSpacing: "-0.045em", fontWeight: "600" },
        ],
        "display-lg": [
          "clamp(2.25rem, 4vw + 0.3rem, 4rem)",
          { lineHeight: "1.05", letterSpacing: "-0.04em", fontWeight: "600" },
        ],
        "display-md": [
          "clamp(1.875rem, 2.8vw + 0.4rem, 3rem)",
          { lineHeight: "1.1", letterSpacing: "-0.035em", fontWeight: "600" },
        ],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(108,130,255,0.18), 0 20px 60px -20px rgba(108,130,255,0.35)",
        soft: "0 1px 0 rgba(255,255,255,0.04), 0 24px 60px -20px rgba(0,0,0,0.45)",
        ring: "inset 0 0 0 1px rgba(255,255,255,0.06)",
      },
      backgroundImage: {
        "grid-fade":
          "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(108,130,255,0.18), transparent 60%)",
        "noise":
          "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.06 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\")",
      },
      animation: {
        "fade-up": "fade-up 0.8s cubic-bezier(0.16, 1, 0.3, 1) both",
        "fade-in": "fade-in 0.8s cubic-bezier(0.16, 1, 0.3, 1) both",
        "shimmer": "shimmer 8s linear infinite",
        "drift": "drift 12s ease-in-out infinite",
        "pulse-soft": "pulse-soft 4s ease-in-out infinite",
        "marquee": "marquee 40s linear infinite",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(14px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        shimmer: {
          "0%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
          "100%": { backgroundPosition: "0% 50%" },
        },
        drift: {
          "0%, 100%": { transform: "translate3d(0,0,0)" },
          "50%": { transform: "translate3d(0,-12px,0)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
        marquee: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
      },
      transitionTimingFunction: {
        "out-expo": "cubic-bezier(0.16, 1, 0.3, 1)",
      },
    },
  },
  plugins: [],
};