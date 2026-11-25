"use client";

import { useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { ChevronLeft, ChevronRight } from "lucide-react";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

export interface Citation {
  chunk_id: string;
  section_title: string;
  page_number: number;
  excerpt: string;
  bboxes_by_page: Record<string, number[][]>;
}

interface PdfViewerProps {
  fileUrl: string | null;
  pageNumber: number;
  onPageChange: (page: number) => void;
  activeCitations: Citation[];
}

export default function PdfViewer({
  fileUrl,
  pageNumber,
  onPageChange,
  activeCitations,
}: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number>(0);

  if (!fileUrl) {
    return (
      <div className="flex h-full items-center justify-center rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 text-sm text-slate-500">
        Upload a PDF to inspect pages and see live AI citation highlights
      </div>
    );
  }

  // Collect all normalized [x0, y0, x1, y1] highlight boxes for the visible page
  const pageBoxes = activeCitations.flatMap(
    (c) => c.bboxes_by_page[String(pageNumber)] || []
  );

  return (
    <div className="flex h-full flex-col rounded-2xl border border-slate-200 bg-slate-100 overflow-hidden">
      {/* Page Flipper Toolbar */}
      <div className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2.5">
        <span className="text-xs font-medium text-slate-600">
          Page {pageNumber} of {numPages || "?"}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPageChange(Math.max(1, pageNumber - 1))}
            disabled={pageNumber <= 1}
            className="rounded-lg border border-slate-200 p-1.5 hover:bg-slate-50 disabled:opacity-40"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button
            onClick={() => onPageChange(Math.min(numPages, pageNumber + 1))}
            disabled={pageNumber >= numPages}
            className="rounded-lg border border-slate-200 p-1.5 hover:bg-slate-50 disabled:opacity-40"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Rendered PDF Page + Highlight Overlay */}
      <div className="flex-1 overflow-auto p-4 flex justify-center">
        <Document
          file={fileUrl}
          onLoadSuccess={({ numPages }) => setNumPages(numPages)}
        >
          <div className="relative shadow-md bg-white">
            <Page pageNumber={pageNumber} width={540} />
            <div className="pointer-events-none absolute inset-0">
              {pageBoxes.map(([x0, y0, x1, y1], i) => (
                <div
                  key={i}
                  className="absolute rounded-sm bg-amber-300/40 ring-1 ring-amber-400/70 transition-all"
                  style={{
                    left: `${x0 * 100}%`,
                    top: `${y0 * 100}%`,
                    width: `${(x1 - x0) * 100}%`,
                    height: `${(y1 - y0) * 100}%`,
                  }}
                />
              ))}
            </div>
          </div>
        </Document>
      </div>
    </div>
  );
}