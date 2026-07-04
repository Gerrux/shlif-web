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
export const IconDownload = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M7 10l5 5 5-5" /><path d="M12 15V3" /></Icon>
);
export const IconEye = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" /><circle cx="12" cy="12" r="3" /></Icon>
);
export const IconEyeOff = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M10.6 6.1A9.7 9.7 0 0 1 12 6c6.5 0 10 6 10 6a17 17 0 0 1-2.7 3.3M6.6 6.6A17 17 0 0 0 2 12s3.5 7 10 7a9.5 9.5 0 0 0 5.4-1.6" /><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2M3 3l18 18" /></Icon>
);
export const IconZoomIn = ({ className }: { className?: string }) => (
  <Icon className={className}><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3M11 8v6M8 11h6" /></Icon>
);
export const IconZoomOut = ({ className }: { className?: string }) => (
  <Icon className={className}><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3M8 11h6" /></Icon>
);
export const IconHand = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M18 11V6a2 2 0 0 0-4 0M14 6V4a2 2 0 0 0-4 0v2M10 6a2 2 0 0 0-4 0v6" /><path d="M18 8a2 2 0 0 1 4 0v6a8 8 0 0 1-8 8h-2a8 8 0 0 1-7.1-4.3L2.7 15a2 2 0 0 1 3.4-2.1" /></Icon>
);
export const IconReset = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3" /></Icon>
);
export const IconBrush = ({ className }: { className?: string }) => (
  <Icon className={className}><path d="M9.06 11.9 3.6 17.4a2.05 2.05 0 0 0 2.9 2.9l5.5-5.46" /><path d="M14 14 21 7a2.1 2.1 0 0 0-3-3l-7 7" /></Icon>
);
