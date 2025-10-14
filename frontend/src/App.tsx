import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useDropzone } from "react-dropzone";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  UploadCloud,
  Image as ImageIcon,
  FileVideo,
  FileText,
  X,
  Download,
  CheckCircle2,
  Info,
  PlayCircle,
  Loader2,
  RefreshCw,
  Copy,
  Check,
  Archive,
  HelpCircle,
  Moon,
  Sun,
} from "lucide-react";

// =====================
// Types for backend
// =====================

type FileType = "text" | "image" | "video" | "archive";

interface BackendOk {
  message: string;
  files: string[];
  processing: any;
}

// =====================
// Config & helpers
// =====================

const API_URL = import.meta?.env?.VITE_UPLOAD_ENDPOINT || "/upload";

// Optional status polling if BE supports it
const API_BASE = (import.meta?.env?.VITE_API_BASE as string) || "";
const STATUS_URL = (jobId: string) => `${API_BASE}/status/${encodeURIComponent(jobId)}`;

const TYPE_STYLES: Record<FileType, { chip: string; accent: string; border: string; label: string }> = {
  image:   { chip: "bg-blue-50 text-blue-900 dark:bg-blue-500/15 dark:text-blue-100",   accent: "bg-blue-500",  border: "border-blue-300 dark:border-blue-500/40", label: "Obrázky" },
  video:   { chip: "bg-green-50 text-green-900 dark:bg-green-500/15 dark:text-green-100",accent: "bg-green-500", border: "border-green-300 dark:border-green-500/40", label: "Video" },
  text:    { chip: "bg-red-50 text-red-900 dark:bg-red-500/15 dark:text-red-100",        accent: "bg-red-500",   border: "border-red-300 dark:border-red-500/40", label: "PDF" },
  archive: { chip: "bg-amber-50 text-amber-900 dark:bg-amber-500/15 dark:text-amber-100",accent: "bg-amber-500", border: "border-amber-300 dark:border-amber-500/40", label: "ZIP" },
};

const EXT_GROUPS: Record<FileType, string[]> = {
  text: [".pdf"],
  image: [".png", ".jpg", ".jpeg", ".webp"],
  video: [".mp4", ".avi", ".mov", ".mkv"],
  archive: [".zip"],
};

function detectType(files: File[]): FileType | null {
  const getGroup = (name: string): FileType | null => {
    const lower = name.toLowerCase();
    if (EXT_GROUPS.image.some((e) => lower.endsWith(e))) return "image";
    if (EXT_GROUPS.video.some((e) => lower.endsWith(e))) return "video";
    if (EXT_GROUPS.text.some((e) => lower.endsWith(e))) return "text";
    if (EXT_GROUPS.archive.some((e) => lower.endsWith(e))) return "archive";
    return null;
  };
  const types = new Set<FileType>();
  for (const f of files) {
    const t = getGroup(f.name);
    if (!t) return null;
    types.add(t);
  }
  if (types.size !== 1) return null;
  return [...types][0];
}

function b64ToBlob(b64: string, contentType: string) {
  const base64 = b64.includes(",") ? b64.split(",")[1] : b64;
  const byteChars = atob(base64);
  const byteNumbers = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) byteNumbers[i] = byteChars.charCodeAt(i);
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: contentType });
}

function downloadText(name: string, text: string, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadXLSX(x: unknown, filename = "output.xlsx") {
  if (typeof x === "string") {
    try {
      const blob = b64ToBlob(x, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      return;
    } catch {}
  }
  downloadText(filename.replace(/\.xlsx$/, ".txt"), String(x));
}

function humanSize(bytes: number) {
  if (!bytes && bytes !== 0) return "";
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, i)).toFixed(1) + " " + ["B", "KB", "MB", "GB"][i];
}

function fileIcon(name: string) {
  const lower = name.toLowerCase();
  const cls = "h-7 w-7"; // bigger icons
  if ([".mp4", ".mov", ".avi", ".mkv"].some((e) => lower.endsWith(e))) return <FileVideo className={cls} />;
  if ([".png", ".jpg", ".jpeg", ".webp"].some((e) => lower.endsWith(e))) return <ImageIcon className={cls} />;
  if (lower.endsWith(".zip")) return <Archive className={cls} />;
  return <FileText className={cls} />;
}

function uploaderBorder(type: FileType | null) {
  if (!type) return "";
  return TYPE_STYLES[type].border;
}

// Copy button
function CopyButton({ getText }: { getText: () => string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs hover:bg-black/10 dark:hover:bg-white/10"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(getText());
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        } catch {}
      }}
      title="Zkopírovat do schránky"
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Zkopírováno" : "Kopírovat"}
    </button>
  );
}

// Backend helpers (video, outputs)
function getOutputs(proc: any): Record<string, unknown> {
  return proc?.outputs || proc?.key_frame_analysis?.outputs || {};
}
function getProcType(proc: any): FileType | undefined {
  return proc?.type;
}
function getTranscript(proc: any): string | null {
  if (!proc) return null;
  return (
    (typeof proc.transcription === "string" && proc.transcription) ||
    (typeof proc.transcript === "string" && proc.transcript) ||
    (typeof proc.asr?.text === "string" && proc.asr.text) ||
    (typeof proc.key_frame_analysis?.transcription === "string" && proc.key_frame_analysis.transcription) ||
    null
  );
}

// Status polling
type StatusPayload = {
  progress: number;
  stage?: string;
  eta_seconds?: number;
  done?: boolean;
  error?: string;
};
async function pollProgress(jobId: string, onTick: (s: StatusPayload) => void, signal: AbortSignal) {
  while (!signal.aborted) {
    try {
      const r = await fetch(STATUS_URL(jobId), { signal, cache: "no-store" });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const s: StatusPayload = await r.json();
      onTick(s);
      if (s.done || s.progress >= 100) break;
    } catch { /* silent backoff */ }
    await new Promise((res) => setTimeout(res, 600));
  }
}

// =====================
// Quick Start Guide
// =====================
function Guide({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      <Card className="relative z-10 max-w-2xl w-full border-white/10 bg-white dark:bg-slate-900 dark:border-slate-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl">Jak používat Media Feature Lab</CardTitle>
          <CardDescription>Krátký průvodce pro první použití</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-relaxed">
          <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
            <b>1) Připrav si soubory</b> — nahraj vždy <i>jeden typ</i> (PDF <b>nebo</b> obrázky <b>nebo</b> video <b>nebo</b> ZIP).
            Do ZIPu vlož jen jeden typ (např. jen obrázky).
          </div>
          <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
            <b>2) Volitelný popis</b> — napiš kontext, co tě zajímá. Pomáhá to modelu vybrat lepší atributy.
          </div>
          <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
            <b>3) Formáty výstupu</b> — vyber JSON/CSV/XLSX/XML podle toho, co chceš stáhnout.
          </div>
          <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
            <b>4) Spusť extrakci</b> — sleduj reálný postup (pokud je dostupný), jinak plynulý odhadovaný progress.
          </div>
          <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
            <b>5) Stáhni výsledky</b> — v sekci <i>Downloads</i> máš velká tlačítka pro export. U videa je i <i>Transcript</i>.
          </div>
          <div className="text-[13px] opacity-80">
            Tip: Pokud vidíš chybu „nelze míchat typy“, odstraň soubory jiného typu nebo je nahraj v dalším kroku.
          </div>
          <div className="pt-2">
            <Button onClick={onClose} className="w-full">Rozumím</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// =====================
// Component
// =====================

export default function MediaFeatureLabPro() {
  const [files, setFiles] = useState<File[]>([]);
  const [fileType, setFileType] = useState<FileType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [response, setResponse] = useState<BackendOk | null>(null);

  const [description, setDescription] = useState("");
  const [formats, setFormats] = useState<{ json: boolean; csv: boolean; xlsx: boolean; xml: boolean }>({
    json: true,
    csv: false,
    xlsx: false,
    xml: false,
  });

  // Subtle theme toggle (not a big button)
  const [deluxe, setDeluxe] = useState(true);
  const [showGuide, setShowGuide] = useState(true);

  // Progress + stepper
  const [progress, setProgress] = useState(0);
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [progressLabel, setProgressLabel] = useState<string>("");

  // Cancellation & animation
  const [uploadCtrl, setUploadCtrl] = useState<AbortController | null>(null);
  const animTimerRef = useRef<number | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    setError(null);
    setResponse(null);
    if (!accepted.length) return;
    const t = detectType(accepted);
    if (!t) {
      setFiles([]);
      setFileType(null);
      setError("Nahraj pouze jeden typ souborů najednou (PDF NEBO obrázky NEBO video NEBO ZIP) a správné přípony.");
      return;
    }
    setFiles(accepted);
    setFileType(t);
    setStep(1);
    setProgress(0);
    setProgressLabel("");
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: true,
    accept: {
      "application/pdf": [".pdf"],
      "image/*": [".png", ".jpg", ".jpeg", ".webp"],
      "video/*": [".mp4", ".avi", ".mov", ".mkv"],
      "application/zip": [".zip"],
    },
  });

  const outputFormatsString = useMemo(() => {
    const selected: string[] = [];
    if (formats.json) selected.push("json");
    if (formats.csv) selected.push("csv");
    if (formats.xlsx) selected.push("xlsx");
    if (formats.xml) selected.push("xml");
    return selected.join(",");
  }, [formats]);

  function labelFor(p: number, formatsLabel: string) {
    if (p < 15) return "Inicializuji…";
    if (p < 35) return "Nahrávám soubory…";
    if (p < 55) return "Validuji typy & metadata…";
    if (p < 75) return "Extrakce featur (LLM/Vision)…";
    if (p < 95) return `Generuji výstupy (${formatsLabel || "JSON"})…`;
    return "Hotovo";
  }

  function startProgressAnimation(maxBeforeFinish = 88, formatsLabel = "JSON") {
    if (animTimerRef.current) window.clearInterval(animTimerRef.current);
    const id = window.setInterval(() => {
      setProgress((p) => {
        const next = p < maxBeforeFinish ? Math.min(maxBeforeFinish, p + Math.random() * 6 + 1) : p;
        setProgressLabel(labelFor(next, formatsLabel));
        return next;
      });
    }, 320);
    animTimerRef.current = id as any;
  }

  function stopProgressAnimation() {
    if (animTimerRef.current) {
      window.clearInterval(animTimerRef.current);
      animTimerRef.current = null;
    }
  }

  useEffect(() => {
    return () => {
      if (animTimerRef.current) window.clearInterval(animTimerRef.current);
      if (uploadCtrl) uploadCtrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleUpload() {
    if (!files.length) return setError("Nejdřív nahraj soubory.");
    if (!fileType) return setError("Detekce typu selhala – smíchal jsi různé přípony?");

    if (uploadCtrl) uploadCtrl.abort();
    stopProgressAnimation();

    const ctrl = new AbortController();
    setUploadCtrl(ctrl);

    setBusy(true);
    setError(null);
    setResponse(null);
    setStep(2);

    const fmtLabel = outputFormatsString.toUpperCase().replaceAll(",", "/") || "JSON";
    setProgress(10);
    setProgressLabel(labelFor(10, fmtLabel));
    startProgressAnimation(88, fmtLabel);

    try {
      const form = new FormData();
      for (const f of files) form.append("files", f, f.name);
      if (outputFormatsString) form.append("output_formats", outputFormatsString);
      if (description.trim()) form.append("description", description.trim());

      const res = await fetch(API_URL, { method: "POST", body: form, signal: ctrl.signal });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `${res.status} ${res.statusText}`);
      }
      const json = (await res.json()) as BackendOk;

      // Real progress via job_id
      const jobId = (json as any)?.processing?.job_id as string | undefined;
      if (jobId) {
        stopProgressAnimation();
        const progCtrl = new AbortController();
        setUploadCtrl(progCtrl);
        await pollProgress(
          jobId,
          (s) => {
            const p = Math.max(0, Math.min(100, s.progress ?? 0));
            setProgress(p);
            const stage = s.stage || labelFor(p, fmtLabel);
            const eta = s.eta_seconds != null ? ` · zbývá ~${Math.ceil(s.eta_seconds)}s` : "";
            setProgressLabel(`${stage}${eta}`);
          },
          progCtrl.signal
        );
      }

      stopProgressAnimation();
      setProgress(100);
      setProgressLabel("Hotovo");
      setResponse(json);
      setStep(3);
    } catch (e: any) {
      stopProgressAnimation();
      if (e?.name !== "AbortError") {
        setError(e?.message || "Nahrání selhalo");
        setStep(1);
      }
      setProgress(0);
      setProgressLabel("");
    } finally {
      setBusy(false);
      setUploadCtrl(null);
      setTimeout(() => setProgress((p) => (p === 100 ? 0 : p)), 1200);
    }
  }

  function handleReset() {
    stopProgressAnimation();
    if (uploadCtrl) uploadCtrl.abort();
    setUploadCtrl(null);
    setBusy(false);
    setFiles([]);
    setFileType(null);
    setError(null);
    setResponse(null);
    setProgress(0);
    setProgressLabel("");
    setStep(1);
  }

  const outputs: Record<string, unknown> = useMemo(() => getOutputs(response?.processing), [response]);
  const procType: FileType | undefined = getProcType(response?.processing);
  const transcript = useMemo(() => getTranscript(response?.processing), [response]);

  const [previews, setPreviews] = useState<string[]>([]);
  useEffect(() => {
    previews.forEach((u) => URL.revokeObjectURL(u));
    const next = files
      .filter((f) => f.type.startsWith("image/") || f.type.startsWith("video/"))
      .map((f) => URL.createObjectURL(f));
    setPreviews(next);
    return () => next.forEach((u) => URL.revokeObjectURL(u));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files]);

  // Default tab preference: show "downloads" first if outputs exist
  const defaultTab = useMemo<"overview" | "details" | "downloads" | "image" | "video">(
    () => (Object.keys(outputs).length ? "downloads" : "overview"),
    [outputs]
  );

  return (
    <div className={`min-h-screen relative overflow-hidden ${deluxe ? "bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800" : "bg-slate-50"}`}>
      {/* background glow */}
      {deluxe && (
        <>
          <div className="pointer-events-none absolute -top-40 -left-40 h-80 w-80 rounded-full bg-indigo-600/20 blur-3xl" />
          <div className="pointer-events-none absolute -bottom-40 -right-40 h-80 w-80 rounded-full bg-fuchsia-600/20 blur-3xl" />
        </>
      )}

      <div className="mx-auto max-w-7xl px-6 py-6">
        {/* Header */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <a href="https://www.vse.cz/" target="_blank" rel="noopener noreferrer">
              <img src="/VSE_logo_CZ_circle_blue.png" alt="Logo školy" className="h-12 w-12 rounded-full shadow" />
            </a>
            <div>
              <h1 className={`text-[28px] font-semibold tracking-tight ${deluxe ? "text-white" : "text-slate-900"}`}>
                Media Feature Lab — Pro
              </h1>
              <p className={`mt-0.5 text-sm ${deluxe ? "text-slate-300" : "text-slate-600"}`}>Prague University of Economics and Business</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={`${deluxe ? "bg-white/10 text-white" : ""}`}>MVP</Badge>

            {/* subtle theme toggle */}
            <Button variant={deluxe ? "secondary" : "outline"} size="icon" className="rounded-full" onClick={() => setDeluxe((v) => !v)} title="Přepnout motiv">
              {deluxe ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>

            {/* help / guide */}
            <Button variant={deluxe ? "secondary" : "default"} size="icon" className="rounded-full" onClick={() => setShowGuide(true)} title="Průvodce">
              <HelpCircle className="h-4 w-4" />
            </Button>

            <Button variant="outline" onClick={handleReset}>
              <RefreshCw className="mr-2 h-4 w-4" /> Reset
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          {/* Uploader */}
          <Card className={`${deluxe ? "bg-white/5 border-white/10" : ""} xl:col-span-2 backdrop-blur-xl`}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className={`text-xl ${deluxe ? "text-white" : ""}`}>Vstupní média</CardTitle>
                  <CardDescription className={`${deluxe ? "text-slate-300" : ""}`}>
                    Nahraj více souborů stejného typu. Backend nepovoluje mix typů v jednom requestu.
                  </CardDescription>
                </div>
                {fileType && (
                  <div className={`px-2 py-1 rounded-lg text-xs font-medium ${TYPE_STYLES[fileType].chip} border ${TYPE_STYLES[fileType].border}`}>
                    {TYPE_STYLES[fileType].label}
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <motion.div whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}>
                <div
                  {...getRootProps()}
                  className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-8 transition
                    ${isDragActive
                      ? `${deluxe ? "border-fuchsia-400/60 bg-fuchsia-500/10" : "border-indigo-500 bg-indigo-50"}`
                      : `${deluxe ? "border-white/10 hover:bg-white/5" : "border-slate-300 hover:bg-slate-50"}`
                    } ${uploaderBorder(fileType)}`}
                >
                  <input {...getInputProps()} />
                  <div className="flex flex-col items-center gap-3">
                    <div className={`rounded-2xl ${deluxe ? "bg-white/10" : "bg-white"} p-4 shadow-sm`}>
                      <UploadCloud className={`${deluxe ? "text-white" : "text-slate-800"} h-9 w-9`} />
                    </div>
                    <p className={`text-[15px] ${deluxe ? "text-slate-200" : "text-slate-700"} font-medium`}>Přetáhni soubory nebo klikni pro výběr</p>
                    <p className={`text-xs ${deluxe ? "text-slate-400" : "text-slate-500"}`}>PDF / JPG / PNG / WEBP / MP4 / MOV / AVI / MKV / ZIP</p>
                  </div>
                </div>
              </motion.div>

              {error && (
                <div className={`mt-4 rounded-xl p-3 text-sm ${deluxe ? "bg-red-400/10 text-red-200" : "bg-red-50 text-red-700"}`}>
                  {error}
                </div>
              )}

              {files.length > 0 && (
                <div className="mt-4">
                  <div className={`mb-2 text-xs ${deluxe ? "text-slate-300" : "text-slate-600"}`}>Vybráno {files.length} souborů</div>
                  <div className="flex flex-wrap gap-2">
                    {files.map((f, i) => (
                      <div
                        key={i}
                        className={`group flex items-center gap-2 rounded-xl border px-3 py-2 ${deluxe ? "bg-white/5 text-white" : "bg-white"}
                         ${fileType ? TYPE_STYLES[fileType].chip : ""} ${fileType ? TYPE_STYLES[fileType].border : ""}`}
                      >
                        <span className={`inline-block h-6 w-1 rounded ${fileType ? TYPE_STYLES[fileType].accent : "bg-slate-300"}`} />
                        <span className="opacity-90">{fileIcon(f.name)}</span>
                        <span className="text-[13px] truncate max-w-[28ch]">{f.name}</span>
                        <span className={`text-[11px] ${deluxe ? "text-slate-300" : "text-slate-600"}`}>{humanSize(f.size)}</span>
                        <button
                          className={`ml-1 rounded-md p-1 ${deluxe ? "hover:bg-white/10" : "hover:bg-slate-100"}`}
                          onClick={() => setFiles((arr) => arr.filter((_, idx) => idx !== i))}
                          title="Odebrat"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Controls */}
              <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
                <div className="space-y-2 md:col-span-2">
                  <Label className={`${deluxe ? "text-slate-200" : ""}`}>Popis (volitelně)</Label>
                  <textarea
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows={3}
                    className={`w-full rounded-xl border p-3 text-[13px] ${deluxe ? "bg-white/5 border-white/10 text-white placeholder:text-slate-400" : ""}`}
                    placeholder="Kontext datasetu, co má LLM zohlednit…"
                  />
                </div>
                <div className="space-y-2">
                  <Label className={`${deluxe ? "text-slate-200" : ""}`}>Formáty výstupu</Label>
                  <div className="flex flex-wrap gap-3 text-sm">
                    {(["json", "csv", "xlsx", "xml"] as const).map((k) => (
                      <label key={k} className={`inline-flex items-center gap-2 rounded-xl px-2 py-1 ${deluxe ? "bg-white/5" : "bg-slate-100"}`}>
                        <input
                          type="checkbox"
                          checked={(formats as any)[k]}
                          onChange={(e) => setFormats((s) => ({ ...s, [k]: e.target.checked }))}
                        />
                        {k.toUpperCase()}
                      </label>
                    ))}
                  </div>
                  <p className={`text-xs ${deluxe ? "text-slate-400" : "text-slate-500"}`}>
                    Backend očekává <code>output_formats</code> jako comma-separated string (např. <code>json,csv</code>).
                  </p>
                </div>
              </div>

              {/* Stepper + CTA */}
              <div className="mt-5 flex flex-wrap items-center gap-3">
                <Button onClick={handleUpload} disabled={!files.length || !!error || busy} className="rounded-xl">
                  {busy ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Zpracovávám…</>) : (<><PlayCircle className="mr-2 h-4 w-4" /> Spustit extrakci</>)}
                </Button>

                <div className={`flex items-center gap-2 text-xs ${deluxe ? "text-slate-300" : "text-slate-500"}`}>
                  <span className={step >= 1 ? "font-medium" : "opacity-60"}>Krok 1: Nahrát</span>
                  <span>→</span>
                  <span className={step >= 2 ? "font-medium" : "opacity-60"}>Krok 2: Extrakce</span>
                  <span>→</span>
                  <span className={step >= 3 ? "font-medium" : "opacity-60"}>Krok 3: Export</span>
                </div>
              </div>

              {(busy || progress > 0) && (
                <div className="mt-4">
                  <div className="relative">
                    <Progress value={progress} />
                    <div className="pointer-events-none absolute -top-6 right-0 rounded-md px-2 py-0.5 text-xs font-medium bg-black/10 backdrop-blur text-slate-800 dark:text-white">
                      {Math.round(progress)}%
                    </div>
                  </div>
                  <p className={`mt-1 text-xs ${deluxe ? "text-slate-300" : "text-slate-500"}`}>{progressLabel}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quick Export (prominent) */}
          <div className="xl:col-span-1">
            <Card className={`${deluxe ? "bg-white/5 border-white/10" : ""} backdrop-blur-xl`}>
              <CardHeader className="pb-3">
                <CardTitle className={`${deluxe ? "text-white" : ""} text-lg`}>Quick Export</CardTitle>
                <CardDescription className={`${deluxe ? "text-slate-300" : ""}`}>Nejdůležitější výstupy na jedno kliknutí</CardDescription>
              </CardHeader>
              <CardContent>
                {Object.keys(outputs).length === 0 ? (
                  <p className={`${deluxe ? "text-slate-300" : "text-slate-600"} text-sm`}>Zatím žádné soubory k dispozici.</p>
                ) : (
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(outputs).map(([k, v]) => (
                      <Button
                        key={k}
                        variant={deluxe ? "secondary" as any : "default"}
                        className={`h-11 justify-start rounded-xl ${deluxe ? "bg-white/10 text-white hover:bg-white/20" : ""}`}
                        onClick={() => {
                          if (k === "json") {
                            const text = typeof v === "string" ? v : JSON.stringify(v, null, 2);
                            downloadText("output.json", text, "application/json;charset=utf-8");
                          } else if (k === "csv") {
                            downloadText("output.csv", String(v), "text/csv;charset=utf-8");
                          } else if (k === "xlsx") {
                            downloadXLSX(v, "output.xlsx");
                          } else if (k === "xml") {
                            downloadText("output.xml", String(v), "application/xml;charset=utf-8");
                          } else if (k === "srt") {
                            downloadText("transcript.srt", String(v), "text/plain;charset=utf-8");
                          } else if (k === "vtt") {
                            downloadText("transcript.vtt", String(v), "text/vtt;charset=utf-8");
                          } else {
                            downloadText(`${k}.txt`, String(v));
                          }
                        }}
                      >
                        <Download className="mr-2 h-4 w-4" />
                        <span className="font-medium">{k.toUpperCase()}</span>
                      </Button>
                    ))}
                  </div>
                )}

                {/* Transcript shortcut if available */}
                {transcript && (
                  <div className="mt-3">
                    <Label className={`${deluxe ? "text-slate-200" : ""}`}>Rychlý náhled transkriptu</Label>
                    <div className={`mt-1 max-h-36 overflow-auto rounded-lg p-3 text-[12px] leading-5 ${deluxe ? "bg-black/30 text-slate-100" : "bg-slate-100"}`}>
                      {transcript}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Results */}
        {response && (
          <div className="mt-6">
            <Card className={`${deluxe ? "bg-white/5 border-white/10" : ""} backdrop-blur-xl`}>
              <CardHeader>
                <CardTitle className={`${deluxe ? "text-white" : ""}`}>Výsledky z backendu</CardTitle>
                <CardDescription className={`${deluxe ? "text-slate-300" : ""}`}>{response.message}</CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue={defaultTab}>
                  <TabsList className={`${deluxe ? "bg-white/10" : ""}`}>
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="details">Processing JSON</TabsTrigger>
                    <TabsTrigger value="downloads">Downloads</TabsTrigger>
                    {procType === "image" && <TabsTrigger value="image">Image Features</TabsTrigger>}
                    {procType === "video" && <TabsTrigger value="video">Video Transcript</TabsTrigger>}
                  </TabsList>

                  <TabsContent value="overview" className="mt-4 space-y-4">
                    <div className={`rounded-xl p-4 ${deluxe ? "bg-white/5 border border-white/10" : "bg-slate-50"}`}>
                      <div className={`text-sm ${deluxe ? "text-slate-200" : "text-slate-700"}`}>
                        <div><span className="opacity-70">Uložené soubory:</span> {response.files?.join(", ")}</div>
                        <div className="mt-1"><span className="opacity-70">Typ zpracování:</span> <b className="capitalize">{procType || "?"}</b></div>
                        {response.processing?.description && (
                          <div className="mt-1"><span className="opacity-70">Popis:</span> {String(response.processing.description)}</div>
                        )}
                        {response.processing?.status && (
                          <div className="mt-1"><span className="opacity-70">Status:</span> {String(response.processing.status)}</div>
                        )}
                      </div>
                    </div>

                    {previews.length > 0 && (
                      <div>
                        <p className={`mb-2 text-sm ${deluxe ? "text-slate-300" : "text-slate-700"}`}>Náhledy</p>
                        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
                          {previews.map((src, idx) => (
                            files[idx]?.type.startsWith("video/") ? (
                              <video key={idx} src={src} className="h-28 w-full rounded-xl object-cover" muted loop autoPlay />
                            ) : (
                              <img key={idx} src={src} className="h-28 w-full rounded-xl object-cover" />
                            )
                          ))}
                        </div>
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="details" className="mt-4">
                    <div className="mb-2 flex items-center justify-between">
                      <span className={`text-sm ${deluxe ? "text-slate-300" : "text-slate-700"}`}>Processing JSON</span>
                      <CopyButton getText={() => JSON.stringify(response.processing, null, 2)} />
                    </div>
                    <pre className={`max-h-[420px] overflow-auto rounded-xl p-4 text-xs ${deluxe ? "bg-black/30 text-slate-100" : "bg-slate-100"}`}>
                      {JSON.stringify(response.processing, null, 2)}
                    </pre>
                  </TabsContent>

                  <TabsContent value="downloads" className="mt-4">
                    {Object.keys(outputs).length === 0 ? (
                      <p className={`${deluxe ? "text-slate-300" : "text-slate-600"} text-sm`}>Žádné soubory k dispozici.</p>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(outputs).map(([k, v]) => (
                          <Button
                            key={k}
                            variant={deluxe ? "secondary" as any : "default"}
                            className={`rounded-xl ${deluxe ? "bg-white/10 text-white hover:bg-white/20" : ""}`}
                            onClick={() => {
                              if (k === "json") {
                                const text = typeof v === "string" ? v : JSON.stringify(v, null, 2);
                                downloadText("output.json", text, "application/json;charset=utf-8");
                              } else if (k === "csv") {
                                downloadText("output.csv", String(v), "text/csv;charset=utf-8");
                              } else if (k === "xlsx") {
                                downloadXLSX(v, "output.xlsx");
                              } else if (k === "xml") {
                                downloadText("output.xml", String(v), "application/xml;charset=utf-8");
                              } else if (k === "srt") {
                                downloadText("transcript.srt", String(v), "text/plain;charset=utf-8");
                              } else if (k === "vtt") {
                                downloadText("transcript.vtt", String(v), "text/vtt;charset=utf-8");
                              } else {
                                downloadText(`${k}.txt`, String(v));
                              }
                            }}
                          >
                            <Download className="mr-2 h-4 w-4" /> {k.toUpperCase()}
                          </Button>
                        ))}
                      </div>
                    )}
                  </TabsContent>

                  {procType === "image" && (
                    <TabsContent value="image" className="mt-4 space-y-4">
                      {response.processing?.feature_specification ? (
                        <div>
                          <div className="mb-2 flex items-center justify-between">
                            <p className={`text-sm ${deluxe ? "text-slate-300" : "text-slate-700"}`}>Feature specification (echo)</p>
                            <CopyButton
                              getText={() =>
                                typeof response.processing.feature_specification === "string"
                                  ? response.processing.feature_specification
                                  : JSON.stringify(response.processing.feature_specification, null, 2)
                              }
                            />
                          </div>
                          <pre className={`max-h-[360px] overflow-auto rounded-xl p-4 text-xs ${deluxe ? "bg-black/30 text-slate-100" : "bg-slate-100"}`}>
                            {typeof response.processing.feature_specification === "string"
                              ? response.processing.feature_specification
                              : JSON.stringify(response.processing.feature_specification, null, 2)}
                          </pre>
                        </div>
                      ) : (
                        <p className={`${deluxe ? "text-slate-300" : "text-slate-600"} text-sm`}>Chybí feature_specification.</p>
                      )}

                      {response.processing?.tabular_output ? (
                        <div>
                          <div className="mb-2 flex items-center justify-between">
                            <p className={`text-sm ${deluxe ? "text-slate-300" : "text-slate-700"}`}>Tabular output</p>
                            <CopyButton
                              getText={() =>
                                typeof response.processing.tabular_output === "string"
                                  ? response.processing.tabular_output
                                  : JSON.stringify(response.processing.tabular_output, null, 2)
                              }
                            />
                          </div>
                          <pre className={`max-h-[360px] overflow-auto rounded-xl p-4 text-xs ${deluxe ? "bg-black/30 text-slate-100" : "bg-slate-100"}`}>
                            {typeof response.processing.tabular_output === "string"
                              ? response.processing.tabular_output
                              : JSON.stringify(response.processing.tabular_output, null, 2)}
                          </pre>
                        </div>
                      ) : null}
                    </TabsContent>
                  )}

                  {procType === "video" && (
                    <TabsContent value="video" className="mt-4 space-y-4">
                      <div className="mb-2 flex items-center justify-between">
                        <p className={`text-sm ${deluxe ? "text-slate-300" : "text-slate-700"}`}>ASR Transkript</p>
                        <CopyButton getText={() => String(transcript || "")} />
                      </div>
                      {transcript ? (
                        <pre className={`max-h-[360px] whitespace-pre-wrap overflow-auto rounded-xl p-4 text-xs ${deluxe ? "bg-black/30 text-slate-100" : "bg-slate-100"}`}>
                          {transcript}
                        </pre>
                      ) : (
                        <p className={`${deluxe ? "text-slate-300" : "text-slate-600"} text-sm`}>
                          Transkript nebyl vrácen. Pokud má být k dispozici, přidej na backendu <code>transcribe_video_file(...)</code>
                          do <code>process_video_files</code> a ulož text do <code>processing.transcription</code>.
                        </p>
                      )}
                    </TabsContent>
                  )}
                </Tabs>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Stav & tipy – kompaktní a dole */}
        <div className="mt-8">
          <Card className={`${deluxe ? "bg-white/5 border-white/10 text-white" : ""} backdrop-blur-xl`}>
            <CardHeader className="py-3">
              <CardTitle className={`text-base ${deluxe ? "text-white" : ""}`}>Stav & tipy</CardTitle>
              <CardDescription className={`${deluxe ? "text-slate-300" : ""} text-xs`}>
                Rychlé rady k formátu odpovědi backendu a častým chybám.
              </CardDescription>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4 opacity-70" />
                <span>Odpověď obsahuje <code>message</code>, <code>files</code>, <code>processing</code>.</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 opacity-70" />
                <span>Nemíchej typy: PDF vs. Images vs. Video vs. ZIP.</span>
              </div>
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4 opacity-70" />
                <span>Pro reálný postup může backend vracet <code>job_id</code> a endpoint <code>/status/&lt;job_id&gt;</code> (progress, stage, ETA).</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Footer */}
        <footer className="mt-10">
          <div className={`h-px w-full ${deluxe ? "bg-gradient-to-r from-fuchsia-500/0 via-fuchsia-500/40 to-fuchsia-500/0" : "bg-gradient-to-r from-indigo-500/0 via-indigo-500/50 to-indigo-500/0"}`} />
          <div className="mt-5 flex flex-col items-center gap-2 text-center">
            <a href="https://www.vse.cz/" target="_blank" rel="noopener noreferrer">
              <img src="/VSE_logo_CZ_circle_blue.png" alt="Logo školy" className={`h-10 w-10 ${deluxe ? "opacity-90" : "opacity-80"}`} />
            </a>
            <p className={`${deluxe ? "text-slate-300" : "text-slate-600"} text-sm`}>
              &copy; {new Date().getFullYear()} Prague University of Economics and Business · Team 2 – Media Feature Lab
            </p>
          </div>
        </footer>
      </div>

      {/* Guide overlay */}
      <Guide open={showGuide} onClose={() => setShowGuide(false)} />
    </div>
  );
}
