import "./global.css"
import { ZenMono } from "@hanzo/font/mono"
import { ZenSans } from "@hanzo/font/sans"
import { RootProvider } from "fumadocs-ui/provider/next"
import type { ReactNode } from "react"

export const metadata = {
  title: {
    default: "Zen zen-eco Documentation",
    template: "%s | Zen zen-eco",
  },
  description:
    "Zen LM zen-eco - Democratizing AI supporting Chain, DAG, and Post-Quantum consensus algorithms.",
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${ZenSans.variable} ${ZenMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-svh bg-background font-sans antialiased">
        <RootProvider
          search={{
            enabled: true,
          }}
          theme={{
            enabled: true,
            defaultTheme: "dark",
          }}
        >
          <div className="relative flex min-h-svh flex-col bg-background">
            {children}
          </div>
        </RootProvider>
      </body>
    </html>
  )
}
