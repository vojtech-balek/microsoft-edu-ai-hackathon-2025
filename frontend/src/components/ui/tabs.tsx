import React, { useState, createContext, useContext } from "react";

type TabsCtx = { value: string; setValue: (v: string) => void };
const Ctx = createContext<TabsCtx | null>(null);

export function Tabs({ defaultValue, children }: { defaultValue: string; children: React.ReactNode }) {
  const [value, setValue] = useState(defaultValue);
  return <Ctx.Provider value={{ value, setValue }}>{children}</Ctx.Provider>;
}

export function TabsList({ children }: { children: React.ReactNode }) {
  return <div className="inline-flex rounded-2xl border bg-white p-1">{children}</div>;
}

export function TabsTrigger({ value, children }: { value: string; children: React.ReactNode }) {
  const ctx = useContext(Ctx)!;
  const active = ctx.value === value;
  return (
    <button
      onClick={() => ctx.setValue(value)}
      className={`px-3 py-1.5 text-sm rounded-xl ${active ? "bg-indigo-600 text-white" : "text-slate-700 hover:bg-slate-100"}`}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value, className = "", children }: { value: string; className?: string; children: React.ReactNode }) {
  const ctx = useContext(Ctx)!;
  if (ctx.value !== value) return null;
  return <div className={className}>{children}</div>;
}
