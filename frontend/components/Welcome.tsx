"use client";
// Полноэкранный welcome команды DATA FORCE. Тёмный микрофото-фон (принцип ДС: сцена тёмная),
// имя команды текстом + эмодзи-ракета, дроп-зона загрузки (drag&drop или клик). Режим
// (крупный план / панорама) здесь не выбирается — переключается уже в рабочей зоне.
import { useState } from "react";
import { IconUpload, IconTelegram } from "@/components/icons";

const TEAM = [
  { name: "Илья", url: "https://t.me/gerrux" },
  { name: "Никита", url: "https://t.me/sngflu" },
];

export function Welcome({ onFile }: { onFile: (f: File) => void }) {
  const [drag, setDrag] = useState(false);

  function pickFromInput(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) onFile(f);
  }
  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  }

  return (
    <section className="welcome full" aria-label="DATA FORCE — классификация руд">
      <div className="welcome-bg" />
      <div className="welcome-scrim" />
      <div className="welcome-inner">
        <h1 className="welcome-title">
          <span className="wm">DATA&nbsp;FORCE</span>
          <span className="rocket">🚀</span>
        </h1>
        <p className="welcome-tagline">Скажи мне, кто твой шлиф</p>
        <div className="welcome-desc">
          <p>
            Сервис определяет сорт руды по снимкам шлифов: рядовая, труднообогатимая или оталькованная.
            Геолог загружает снимок и получает вердикт за секунды вместо часов ручной работы.
          </p>
          <p>
            Сначала выравниваем освещение, потом нейросеть отделяет руду от породы, а классификатор
            определяет сорт по размеру зерен сульфидов — крупные дают рядовую руду, мелкие труднообогатимую.
            Логику вердикта нам подсказали сами геологи.
          </p>
          <p>
            Точно выделить тальк по одному снимку физически нельзя, поэтому система показывает
            зону-кандидат, а финальное решение остается за экспертом.
          </p>
          <p>
            Любую маску можно поправить прямо в интерфейсе: кисть, ластик, автозаливка — и вердикт
            пересчитывается на месте. Это и есть режим экспертной проверки из ТЗ.
          </p>
          <p>
            На выходе — цветная маска, проценты по фазам, заключение и экспорт в PDF и CSV. Большие
            панорамы обрабатываем по частям, сервис работает и на обычном компьютере, и на сервере
            с видеокартой.
          </p>
        </div>
        <label
          className={`dropzone${drag ? " drag" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragEnter={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
        >
          <IconUpload className="ico-lg dz-ico" />
          <span className="dz-title">Перетащите снимок шлифа сюда</span>
          <span className="dz-sub">или нажмите, чтобы выбрать · JPG / PNG · OM, отражённый свет</span>
          <input type="file" accept="image/*" onChange={pickFromInput} style={{ display: "none" }} />
        </label>
      </div>
      <div className="welcome-credits">
        <span className="wc-label">Команда</span>
        <div className="wc-links">
          {TEAM.map((m) => (
            <a key={m.url} className="wc-link" href={m.url} target="_blank" rel="noopener noreferrer">
              <IconTelegram className="ico-sm" />{m.name}
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
