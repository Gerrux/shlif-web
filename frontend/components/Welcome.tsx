"use client";
// Welcome-герой команды DATA FORCE. Тёмный микрофото-фон (принцип ДС: сцена тёмная),
// имя команды текстом + эмодзи-ракета, сегмент режима и CTA-загрузка запускают анализ.
import type { Mode } from "@/lib/api/types";
import { IconUpload, IconArrow } from "@/components/icons";

const MODES: [Mode, string][] = [["closeup", "Крупный план"], ["panorama", "Панорама"]];

export function Welcome({
  mode, onMode, onFile,
}: {
  mode: Mode;
  onMode: (m: Mode) => void;
  onFile: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <section className="welcome" aria-label="DATA FORCE — классификация руд">
      <div className="welcome-bg" />
      <div className="welcome-scrim" />
      <div className="welcome-inner">
        <div className="welcome-badge"><span className="ping" />ИИ-классификация полированных шлифов</div>
        <h1 className="welcome-title">
          <span className="wm">DATA&nbsp;FORCE</span>
          <span className="rocket">🚀</span>
        </h1>
        <p className="welcome-tagline">Скажи мне, кто твой шлиф</p>
        <p className="welcome-desc">
          Автоматическая классификация руд по панорамным OM-изображениям полированных шлифов:
          сегментация сульфидных фаз, детекция талька и вердикт по экспертной логике.
        </p>
        <div className="welcome-actions">
          <div className="seg on-dark" role="group" aria-label="Режим анализа">
            {MODES.map(([m, label]) => (
              <button key={m} type="button" className={mode === m ? "active" : ""}
                aria-pressed={mode === m} onClick={() => onMode(m)}>{label}</button>
            ))}
          </div>
          <label className="btn primary lg welcome-cta">
            <IconUpload className="ico-md" />
            Загрузить шлиф
            <IconArrow className="ico-md arrow" />
            <input type="file" accept="image/*" onChange={onFile} style={{ display: "none" }} />
          </label>
        </div>
      </div>
    </section>
  );
}
