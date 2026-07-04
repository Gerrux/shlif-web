// Line-иконки дизайн-системы «Шлиф» (stroke 1.8, viewBox 24). Заменяют эмодзи в интерфейсе.
import type { ReactNode } from "react";

function Icon({ className = "ico-sm", sw = 1.8, children }: { className?: string; sw?: number; children: ReactNode }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  );
}

export const IconHex = ({ className }: { className?: string }) => (
  <Icon className={className} sw={1.9}><path d="M12 2 3 7v10l9 5 9-5V7z" /><path d="M12 22V12M3 7l9 5 9-5" /></Icon>
);
export const IconUpload = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M17 8l-5-5-5 5" /><path d="M12 3v13" /></Icon>
);
export const IconEdit = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M11 5H6a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-5" /><path d="M18.5 2.5a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z" /></Icon>
);
export const IconSave = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" /><path d="M17 21v-8H7v8" /><path d="M7 3v5h8" /></Icon>
);
export const IconUndo = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M9 14 4 9l5-5" /><path d="M4 9h11a5 5 0 0 1 0 10H9" /></Icon>
);
export const IconRedo = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="m15 14 5-5-5-5" /><path d="M20 9H9a5 5 0 0 0 0 10h6" /></Icon>
);
export const IconSun = ({ className }: { className?: string }) => (
  <Icon className={className}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></Icon>
);
export const IconMoon = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></Icon>
);
export const IconAlert = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /><path d="M12 9v4M12 17h.01" /></Icon>
);
export const IconScan = ({ className }: { className?: string }) => (
  <Icon className={className}><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" /></Icon>
);
export const IconArrow = ({ className }: { className?: string }) => (
  <Icon className={className} sw={2}><path d="M5 12h14M13 6l6 6-6 6" /></Icon>
);
