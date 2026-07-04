import "./globals.css";
import Providers from "./providers";

export const metadata = {
  title: "Шлиф · классификация руд — DATA FORCE",
  description: "Автоматическая классификация руд по панорамным OM-изображениям полированных шлифов.",
};

// Применяем сохранённую тему до отрисовки, чтобы не было вспышки светлой темы.
const themeInit = `(function(){try{var t=localStorage.getItem('shlif-theme');if(t==='dark'||t==='light'){document.documentElement.setAttribute('data-theme',t);}}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
        />
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
