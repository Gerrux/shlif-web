"use client";
// Переключатель светлой/тёмной темы дизайн-системы. Тема пишется в data-theme на <html>
// и сохраняется в localStorage; предзагрузочный скрипт в layout применяет её без мигания.
import { useEffect, useState } from "react";
import { IconSun, IconMoon } from "@/components/icons";

type Theme = "light" | "dark";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("light");

  useEffect(() => {
    const cur = document.documentElement.getAttribute("data-theme");
    setTheme(cur === "dark" ? "dark" : "light");
  }, []);

  function set(t: Theme) {
    setTheme(t);
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("shlif-theme", t); } catch { /* приватный режим */ }
  }

  return (
    <div className="theme-toggle" role="group" aria-label="Тема оформления">
      <button type="button" className={theme === "light" ? "active" : ""} aria-pressed={theme === "light"} onClick={() => set("light")}>
        <IconSun /> Светлая
      </button>
      <button type="button" className={theme === "dark" ? "active" : ""} aria-pressed={theme === "dark"} onClick={() => set("dark")}>
        <IconMoon /> Тёмная
      </button>
    </div>
  );
}
