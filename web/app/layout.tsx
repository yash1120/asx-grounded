import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "asx-grounded",
  description: "Grounded Q&A over ASX continuous-disclosure announcements with a public eval scoreboard.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans">
        <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
          <Link href="/" className="font-mono text-sm font-semibold">
            asx-grounded
          </Link>
          <nav className="flex gap-4 text-sm text-zinc-400">
            <Link href="/" className="hover:text-zinc-100">Query</Link>
            <Link href="/eval" className="hover:text-zinc-100">Eval scoreboard</Link>
            <a
              href="https://github.com/your-handle/asx-grounded"
              className="hover:text-zinc-100"
              target="_blank"
              rel="noreferrer"
            >
              GitHub
            </a>
          </nav>
        </header>
        <main className="max-w-3xl mx-auto px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
