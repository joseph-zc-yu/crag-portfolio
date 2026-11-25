"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Search, Send, Upload, ExternalLink, FileText } from "lucide-react";
import type { Citation } from "@/components/PdfViewer";

const PdfViewer = dynamic(() => import("@/components/PdfViewer"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface WebResult {
  id: string;
  query: string;
  title: string;
  url: string;
  snippet: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  relevance?: "high" | "insufficient";
  citations?: Citation[];
  web_queries?: string[];
  web_results?: WebResult[];
}

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);
  const [webQueries, setWebQueries] = useState<string[]>([]);
  const [webResults, setWebResults] = useState<WebResult[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setPdfUrl(URL.createObjectURL(file));
    setPageNumber(1);
    setSessionId(null);
    setActiveCitations([]);
    setWebQueries([]);
    setWebResults([]);
    setMessages([
      {
        role: "assistant",
        content: `Indexing **${file.name}**... (First upload may take ~40s while Render wakes the backend.)`,
      },
    ]);

    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.session_id) {
        throw new Error(data.detail || `Upload failed (HTTP ${res.status})`);
      }
      setSessionId(data.session_id);
      setMessages([
        {
          role: "assistant",
          content: `Indexed **${data.filename}** (${data.chunk_count} sections). Ask a question below!`,
        },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Network or CORS error.";
      setMessages([
        {
          role: "assistant",
          content: `Upload error: ${msg}\n(Backend URL: ${API_BASE})`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || !sessionId || loading) return;

    const query = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, query }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Chat failed (HTTP ${res.status})`);
      }

      const citations: Citation[] = data.citations || [];
      setActiveCitations(citations);
      if (citations.length > 0) setPageNumber(citations[0].page_number);

      setWebQueries(data.web_queries || []);
      setWebResults(data.web_results || []);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          relevance: data.relevance,
          citations,
          web_queries: data.web_queries,
          web_results: data.web_results,
        },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Request failed.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${msg}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="grid h-screen grid-cols-12 gap-4 bg-white p-4 text-slate-800">
      {/* Left: Interactive PDF Viewer */}
      <section className="col-span-6 flex flex-col gap-3 overflow-hidden">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-semibold tracking-tight">Corrective RAG PDF Agent</h1>
          <label className="flex cursor-pointer items-center gap-2 rounded-xl bg-slate-900 px-3.5 py-2 text-xs font-medium text-white hover:bg-slate-700">
            <Upload className="h-3.5 w-3.5" />
            Upload PDF
            <input type="file" accept="application/pdf" className="hidden" onChange={handleUpload} />
          </label>
        </div>
        <div className="flex-1 overflow-hidden">
          <PdfViewer
            fileUrl={pdfUrl}
            pageNumber={pageNumber}
            onPageChange={setPageNumber}
            activeCitations={activeCitations}
          />
        </div>
      </section>

      {/* Middle: Multi-turn Chat & Inline Citations */}
      <section className="col-span-4 flex flex-col rounded-2xl border border-slate-200 bg-slate-50 overflow-hidden">
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`rounded-2xl p-3.5 text-sm leading-relaxed ${
                m.role === "user"
                  ? "ml-8 bg-slate-900 text-white"
                  : "mr-4 border border-slate-200 bg-white"
              }`}
            >
              {m.relevance && (
                <span
                  className={`mb-2 inline-block rounded-lg px-2 py-0.5 text-[11px] font-medium ${
                    m.relevance === "high"
                      ? "bg-emerald-50 text-emerald-700"
                      : "bg-amber-50 text-amber-700"
                  }`}
                >
                  {m.relevance === "high"
                    ? "Grounded in PDF"
                    : "Supplemented via Web Search"}
                </span>
              )}

              <div className="space-y-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:leading-relaxed [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:pl-5">
                <ReactMarkdown>{m.content}</ReactMarkdown>
              </div>

              {/* Clickable PDF Citation Chips */}
              {m.citations && m.citations.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {m.citations.map((c) => (
                    <button
                      key={c.chunk_id}
                      onClick={() => {
                        setActiveCitations([c]);
                        setPageNumber(c.page_number);
                      }}
                      className="inline-flex items-center gap-1 rounded-lg border border-amber-200 bg-amber-50 px-2 py-1 text-xs font-medium text-amber-900 hover:bg-amber-100"
                    >
                      <FileText className="h-3 w-3" />
                      p.{c.page_number} · {c.section_title.slice(0, 24)}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

          {loading && sessionId && (
            <div className="mr-4 animate-pulse rounded-2xl border border-slate-200 bg-white p-3.5 text-sm text-slate-500">
              Analyzing PDF sections &amp; grading relevance...
            </div>
          )}
        </div>

        <form onSubmit={handleSend} className="flex gap-2 border-t border-slate-200 bg-white p-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={sessionId ? "Ask a follow-up about the PDF..." : "Upload a PDF first..."}
            disabled={!sessionId || loading}
            className="flex-1 rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none focus:border-slate-400"
          />
          <button
            type="submit"
            disabled={!sessionId || loading}
            className="rounded-xl bg-slate-900 px-3.5 py-2 text-white hover:bg-slate-700 disabled:opacity-40"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </section>

      {/* Right Sidebar: Lightly-Rounded Rectangular Web Search Bubbles */}
      <aside className="col-span-2 flex flex-col gap-3 overflow-y-auto rounded-2xl border border-slate-200 bg-slate-50 p-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          External Search
        </h2>

        {webQueries.length === 0 ? (
          <p className="text-xs text-slate-400">
            When the PDF lacks sufficient info, DuckDuckGo search terms &amp; sources appear here.
          </p>
        ) : (
          <>
            <div className="space-y-1.5">
              <span className="text-[11px] font-medium text-slate-500">Queries Used</span>
              <div className="flex flex-wrap gap-1.5">
                {webQueries.map((q, i) => (
                  <div
                    key={i}
                    className="inline-flex items-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-2.5 py-1 text-xs font-medium text-sky-900"
                  >
                    <Search className="h-3 w-3 shrink-0" />
                    <span>{q}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-[11px] font-medium text-slate-500">Web Sources</span>
              {webResults.map((w) => (
                <a
                  key={w.id}
                  href={w.url || "#"}
                  target="_blank"
                  rel="noreferrer"
                  className="block rounded-xl border border-slate-200 bg-white p-2.5 transition hover:border-slate-300"
                >
                  <div className="flex items-start justify-between gap-1">
                    <span className="line-clamp-2 text-xs font-semibold text-slate-800">
                      {w.title}
                    </span>
                    <ExternalLink className="h-3 w-3 shrink-0 text-slate-400" />
                  </div>
                  <p className="mt-1 line-clamp-3 text-[11px] text-slate-500">{w.snippet}</p>
                </a>
              ))}
            </div>
          </>
        )}
      </aside>
    </main>
  );
}