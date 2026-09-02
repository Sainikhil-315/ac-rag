export default function UserMessage({ content }) {
  return (
    <div className="flex justify-end animate-slide-up">
      <div className="max-w-[70%] bg-accent/15 border border-accent/20 rounded-2xl rounded-tr-sm px-4 py-3 text-sm text-primary leading-relaxed">
        {content}
      </div>
    </div>
  )
}
