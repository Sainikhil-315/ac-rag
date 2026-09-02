export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#F4F4F5',
        surface: '#F7F7F8',
        'surface-2': '#EDEDEF',
        border: '#E4E4E7',
        accent: '#5E6AD2',
        'accent-hover': '#4F5BC2',
        primary: '#18181B',
        secondary: '#71717A',
        muted: '#A1A1AA',
        success: '#059669',
        warning: '#D97706',
        danger: '#DC2626',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'blink': 'blink 1s step-end infinite',
        'pulse-dot': 'pulseDot 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: 'translateY(8px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        blink: { '0%,100%': { opacity: 1 }, '50%': { opacity: 0 } },
        pulseDot: { '0%,100%': { transform: 'scale(1)', opacity: 1 }, '50%': { transform: 'scale(1.2)', opacity: 0.7 } },
      }
    }
  },
  plugins: []
}
