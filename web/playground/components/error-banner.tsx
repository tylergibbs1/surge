import { ExclamationTriangleIcon } from "@radix-ui/react-icons"

export function ErrorBanner({
  title,
  detail,
  className = "",
}: {
  title: string
  detail?: string
  className?: string
}) {
  return (
    <div
      role="alert"
      className={`flex items-start gap-2.5 rounded-md border border-destructive/25 bg-destructive/8 px-3 py-2 text-xs ${className}`}
    >
      <ExclamationTriangleIcon
        aria-hidden="true"
        className="mt-[1px] size-3.5 shrink-0 text-destructive"
      />
      <div className="min-w-0 flex-1 space-y-0.5">
        <p className="font-medium text-destructive">{title}</p>
        {detail ? (
          <p className="break-words font-mono tabular-nums text-destructive/80">
            {detail}
          </p>
        ) : null}
      </div>
    </div>
  )
}
