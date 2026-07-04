"use client";
import { useCallback, useEffect, useRef, useState } from "react";

const ACCEPT = "image/*";
const HINT = "JPG · PNG · TIFF";

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} КБ`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;
}

/**
 * Зона загрузки снимка шлифа: перетаскивание файла ИЛИ клик для выбора,
 * с миниатюрой-превью выбранного изображения. Отдаёт выбранный File наверх.
 */
export function DropZone({ onFile, busy = false }: { onFile: (f: File) => void; busy?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [preview, setPreview] = useState<{ url: string; name: string; size: number } | null>(null);

  // Освобождаем предыдущий object URL при смене превью / размонтировании — без утечек.
  useEffect(() => {
    if (!preview) return;
    return () => URL.revokeObjectURL(preview.url);
  }, [preview]);

  const accept = useCallback(
    (f: File | null | undefined) => {
      if (!f || busy) return;
      if (!f.type.startsWith("image/")) return;
      setPreview({ url: URL.createObjectURL(f), name: f.name, size: f.size });
      onFile(f);
    },
    [busy, onFile],
  );

  function openPicker() {
    if (!busy) inputRef.current?.click();
  }
  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openPicker();
    }
  }
  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDrag(false);
    accept(e.dataTransfer.files?.[0]);
  }
  function onDragOver(e: React.DragEvent) {
    e.preventDefault();
    if (!busy) setDrag(true);
  }
  function onDragLeave(e: React.DragEvent) {
    // Игнорируем переходы на вложенные элементы, чтобы подсветка не мигала.
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setDrag(false);
  }

  return (
    <div
      className={`dropzone${drag ? " drag" : ""}${busy ? " busy" : ""}`}
      role="button"
      tabIndex={busy ? -1 : 0}
      aria-label="Загрузить снимок шлифа"
      aria-disabled={busy || undefined}
      onClick={openPicker}
      onKeyDown={onKeyDown}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = ""; // позволяем повторно выбрать тот же файл
        }}
      />
      {preview ? (
        <div className="dz-preview">
          <img src={preview.url} alt="превью" className="dz-thumb" />
          <div className="dz-meta">
            <div className="dz-name">{preview.name}</div>
            <div className="dz-sub">
              {fmtSize(preview.size)} · {busy ? "анализ…" : "готово к анализу"}
            </div>
          </div>
          <button
            type="button"
            className="dz-replace"
            onClick={(e) => {
              e.stopPropagation();
              openPicker();
            }}
          >
            ✕ Заменить
          </button>
        </div>
      ) : (
        <div className="dz-empty">
          <div className="dz-icon">◈</div>
          <div className="dz-title">{drag ? "Отпустите файл" : "Перетащите снимок шлифа сюда"}</div>
          <div className="dz-hint">или нажмите для выбора · {HINT}</div>
        </div>
      )}
    </div>
  );
}
